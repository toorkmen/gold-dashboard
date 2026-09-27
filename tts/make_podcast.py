#!/usr/bin/env python3
"""Build a two-voice Persian podcast MP3 from a dialogue script, read word for word.

Engines
  gemini     Google Gemini TTS (needs GEMINI_API_KEY; free tier has a daily request cap).
  microsoft  Microsoft Read Aloud voices Farid / Dilara through edge-tts (unofficial, no key).
  auto       (default) Gemini first; if it can't finish (no key, daily quota used up, outage)
             the WHOLE podcast is rebuilt with Microsoft, so a run never ends empty-handed.

Script format (same rules as the dashboard's parsePodcastScript):
    آقا: ...   / Man: ... / Host: ...   -> male voice
    خانم: ...  / Woman: ... / Guest: ... -> female voice
    A line without a label continues the previous speaker's turn.
    If the script has no labels at all, the whole text is read with the male voice.

Two ways to run
  * JOB_ID set (dashboard -> Apps Script -> this workflow): fetch the script from Apps Script,
    build the MP3, upload it back to Drive.
  * JOB_ID empty (manual "Run workflow" test on GitHub): read SCRIPT_TEXT ("\\n" = new line) and
    write one MP3 per engine into out/ so the two can be compared.
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

# ---------------- settings ----------------
ENGINE = (os.environ.get("TTS_ENGINE") or "auto").strip().lower()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = [m.strip() for m in (os.environ.get("GEMINI_MODEL") or
                 "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts").split(",") if m.strip()]
GEMINI_MALE = os.environ.get("GEMINI_MALE_VOICE") or "Charon"    # "Informative"
GEMINI_FEMALE = os.environ.get("GEMINI_FEMALE_VOICE") or "Kore"  # "Firm"
GEMINI_STYLE = os.environ.get("GEMINI_STYLE") or "clear, calm and informative podcast narration"
GEMINI_CHUNK_CHARS = int(os.environ.get("GEMINI_CHUNK_CHARS") or "3000")
API = "https://generativelanguage.googleapis.com/v1beta"

MS_MALE = os.environ.get("MALE_VOICE") or "fa-IR-FaridNeural"
MS_FEMALE = os.environ.get("FEMALE_VOICE") or "fa-IR-DilaraNeural"
MS_RATE = os.environ.get("TTS_RATE") or "+0%"
MS_CONCURRENCY = 3

PAUSE_SEC = float(os.environ.get("PAUSE_SEC") or "0.45")
SAMPLE_RATE = 24000

GAS_URL = os.environ.get("GAS_URL", "").strip()
SECRET = os.environ.get("TTS_SECRET", "").strip()
JOB_ID = os.environ.get("JOB_ID", "").strip()

LABEL_RE = re.compile(r"^(آقا|خانم|man|woman|host|guest)\s*:\s*(.*)$", re.IGNORECASE)


class QuotaExhausted(RuntimeError):
    """Gemini's daily free-tier cap is used up — no point retrying today."""


# ---------------- script parsing ----------------
def clean(text):
    # Markdown marks (**bold**, # headings, `code`, > quotes) would otherwise be read aloud.
    text = re.sub(r"[*_`#>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_script(text):
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LABEL_RE.match(line)
        if m:
            raw = m.group(1)
            speaker = "male" if (raw == "آقا" or raw.lower() in ("man", "host")) else "female"
            segments.append([speaker, m.group(2)])
        elif segments:
            segments[-1][1] += " " + line
    segments = [(spk, clean(t)) for spk, t in segments]
    segments = [(spk, t) for spk, t in segments if t]
    if not segments and clean(text):
        segments = [("male", clean(text))]
    return segments


def chunk_turns(segments, max_chars):
    """Group consecutive turns into requests of at most ~max_chars (a single long turn stays whole)."""
    chunks, cur, size = [], [], 0
    for spk, t in segments:
        if cur and size + len(t) > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append((spk, t))
        size += len(t)
    if cur:
        chunks.append(cur)
    return chunks


# ---------------- audio helpers ----------------
def run(cmd):
    subprocess.run(cmd, check=True)


def to_wav(audio_bytes, path, rate=SAMPLE_RATE):
    """Gemini returns WAV (RIFF) for normal requests, raw 16-bit PCM in some cases — normalise to WAV."""
    if audio_bytes[:4] == b"RIFF":
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return
    raw = path + ".pcm"
    with open(raw, "wb") as f:
        f.write(audio_bytes)
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", raw, path])


def join_to_mp3(paths, out_path, workdir):
    """Concatenate segment files (all WAV or all MP3) with a short pause between them, encode one MP3."""
    ext = os.path.splitext(paths[0])[1]
    silence = os.path.join(workdir, "silence" + ext)
    codec = ["-c:a", "pcm_s16le"] if ext == ".wav" else ["-c:a", "libmp3lame", "-b:a", "48k"]
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
         "-t", str(PAUSE_SEC), *codec, silence])
    listfile = os.path.join(workdir, "list.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for i, p in enumerate(paths):
            if i:
                f.write(f"file '{silence}'\n")
            f.write(f"file '{p}'\n")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listfile,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", "64k", out_path])


# ---------------- Gemini ----------------
def find_audio(obj):
    """Return (base64_data, mime) of the LAST audio block anywhere in a Gemini response.
    Works for the Interactions API (steps[].content[] with type=audio) and the legacy
    generateContent API (candidates[].content.parts[].inlineData)."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            mime = str(o.get("mime_type") or o.get("mimeType") or "")
            if isinstance(o.get("data"), str) and len(o["data"]) > 100 and (o.get("type") == "audio" or "audio" in mime):
                found.append((o["data"], mime))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return found[-1] if found else (None, None)


def _gemini_bodies(model, turns):
    names = {"male": "Man", "female": "Woman"}
    interactions = {
        "model": model,
        "input": [{"type": "user_input", "content": [
            {"type": "text", "text": t,
             "annotations": [{"type": "speech_metadata", "speaker": names[s], "style": GEMINI_STYLE}]}
            for s, t in turns]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": {"mode": "conversational", "speakers": [
            {"speaker": "Man", "voice": GEMINI_MALE}, {"speaker": "Woman", "voice": GEMINI_FEMALE}]}},
    }
    legacy = {
        "contents": [{"role": "user", "parts": [
            {"text": t, "speech_metadata": {"speaker": names[s], "style": GEMINI_STYLE}} for s, t in turns]}],
        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"multiSpeakerVoiceConfig": {
            "speakerVoiceConfigs": [
                {"speaker": "Man", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_MALE}}},
                {"speaker": "Woman", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_FEMALE}}}]}}},
    }
    return [("interactions", f"{API}/interactions", interactions),
            ("generateContent", f"{API}/models/{model}:generateContent", legacy)]


def _retry_delay(resp, attempt):
    m = re.search(r'"retryDelay"\s*:\s*"(\d+)', resp.text)
    return min(int(m.group(1)) + 2, 90) if m else 15 * (attempt + 1)


_WORKING = None  # (model, shape) that last succeeded — tried first so later chunks don't waste requests


def gemini_chunk(turns):
    """One request (up to 2 speakers) -> audio bytes. Tries each model and both API shapes."""
    global _WORKING
    headers = {"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}
    errors = []
    candidates = [(model, shape, url, body) for model in GEMINI_MODELS for shape, url, body in _gemini_bodies(model, turns)]
    candidates.sort(key=lambda c: (c[0], c[1]) != _WORKING)  # stable: known-good first, rest keep their order
    for model, shape, url, body in candidates:
        for attempt in range(5):
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=300)
            if resp.status_code == 200:
                data, mime = find_audio(resp.json())
                if data:
                    _WORKING = (model, shape)
                    rate = re.search(r"rate=(\d+)", mime or "")
                    return base64.b64decode(data), int(rate.group(1)) if rate else SAMPLE_RATE
                errors.append(f"{model}/{shape}: 200 but no audio in the response")
                break
            if resp.status_code == 429:
                if re.search(r"per ?day|PerDay|daily", resp.text, re.I):
                    raise QuotaExhausted("Gemini free-tier daily limit reached")
                time.sleep(_retry_delay(resp, attempt))  # per-minute limit: wait and retry
                continue
            if resp.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            errors.append(f"{model}/{shape}: HTTP {resp.status_code} {resp.text[:200]}")
            break  # 400/403/404: try the next API shape / model
        else:
            errors.append(f"{model}/{shape}: still failing after retries")
    raise RuntimeError("Gemini TTS failed — " + " | ".join(errors[-4:]))


def build_gemini(segments, workdir):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    chunks = chunk_turns(segments, GEMINI_CHUNK_CHARS)
    print(f"Gemini: {len(chunks)} request(s), voices {GEMINI_MALE}/{GEMINI_FEMALE}, models {GEMINI_MODELS}")
    paths = []
    for i, turns in enumerate(chunks):
        audio, rate = gemini_chunk(turns)
        p = os.path.join(workdir, f"g_{i:04d}.wav")
        to_wav(audio, p, rate)
        paths.append(p)
        print(f"  chunk {i + 1}/{len(chunks)} done")
    return paths


# ---------------- Microsoft (edge-tts) ----------------
async def _ms_one(text, voice, path):
    import edge_tts
    last = None
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, voice, rate=MS_RATE).save(path)
            if os.path.getsize(path) > 0:
                return
            last = RuntimeError("empty audio returned")
        except Exception as e:  # network hiccups / throttling — retry with backoff
            last = e
        await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Microsoft voice service failed after 4 attempts: {last}")


async def _ms_all(segments, workdir):
    sem = asyncio.Semaphore(MS_CONCURRENCY)
    paths = [os.path.join(workdir, f"m_{i:04d}.mp3") for i in range(len(segments))]

    async def one(i, spk, text):
        async with sem:
            await _ms_one(text, MS_MALE if spk == "male" else MS_FEMALE, paths[i])

    await asyncio.gather(*(one(i, s, t) for i, (s, t) in enumerate(segments)))
    return paths


def build_microsoft(segments, workdir):
    print(f"Microsoft: {len(segments)} turn(s), voices {MS_MALE}/{MS_FEMALE}")
    return asyncio.run(_ms_all(segments, workdir))


# ---------------- orchestration ----------------
def build_mp3(text, engine=ENGINE):
    """Returns (mp3_bytes, engine_actually_used)."""
    segments = parse_script(text)
    if not segments:
        raise RuntimeError("the script is empty")
    male = sum(1 for s, _ in segments if s == "male")
    print(f"{len(segments)} turn(s): {male} male, {len(segments) - male} female, "
          f"{sum(len(t) for _, t in segments)} characters")
    order = {"gemini": ["gemini"], "microsoft": ["microsoft"]}.get(engine, ["gemini", "microsoft"])
    notes = []
    for eng in order:
        with tempfile.TemporaryDirectory() as workdir:
            try:
                paths = build_gemini(segments, workdir) if eng == "gemini" else build_microsoft(segments, workdir)
                out = os.path.join(workdir, "podcast.mp3")
                join_to_mp3(paths, out, workdir)
                with open(out, "rb") as f:
                    data = f.read()
                print(f"MP3 built with {eng}: {len(data) / 1e6:.2f} MB")
                return data, eng
            except Exception as e:
                notes.append(f"{eng}: {e}")
                print(f"{eng} failed: {e}")
    raise RuntimeError("; ".join(notes))


def gas_get(params):
    r = requests.get(GAS_URL, params=params, timeout=90)
    r.raise_for_status()
    return r.json()


def gas_post(data):
    # Apps Script answers a POST with a redirect; the request is already processed by then.
    r = requests.post(GAS_URL, data=data, timeout=300)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return {"status": "unknown"}


def main():
    if JOB_ID:
        if not GAS_URL or not SECRET:
            sys.exit("Repository secrets GAS_URL and TTS_SECRET must both be set.")
        info = gas_get({"action": "ttsjob", "job": JOB_ID, "secret": SECRET})
        if info.get("status") != "ok":
            sys.exit(f"Could not fetch the script from Apps Script: {info.get('message', info.get('status'))}")
        try:
            data, used = build_mp3(info["text"])
        except Exception as e:
            gas_post({"action": "ttserror", "job": JOB_ID, "secret": SECRET, "message": str(e)[:500]})
            raise
        res = gas_post({"action": "ttsresult", "job": JOB_ID, "secret": SECRET, "engine": used,
                        "content": base64.b64encode(data).decode("ascii")})
        # the repo is public, so its Actions logs are too: log the outcome only, never the script text
        print(f"Uploaded to Drive ({used}): {res.get('filename', res.get('status'))}")
    else:
        text = (os.environ.get("SCRIPT_TEXT") or "").replace("\\n", "\n")
        os.makedirs("out", exist_ok=True)
        engines = ["gemini", "microsoft"] if ENGINE == "auto" else [ENGINE]
        made = 0
        for eng in engines:
            try:
                data, _ = build_mp3(text, eng)
                with open(f"out/podcast_{eng}.mp3", "wb") as f:
                    f.write(data)
                made += 1
            except Exception as e:
                print(f"Test with {eng} failed: {e}")
        if not made:
            sys.exit("No engine produced audio — see the errors above.")
        print("Test MP3(s) written to out/ — download them from the run's Artifacts.")


if __name__ == "__main__":
    main()
