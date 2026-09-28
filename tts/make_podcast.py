#!/usr/bin/env python3
"""Build a two-voice Persian podcast MP3 from a dialogue script, read word for word.

Engines
  gemini     Google Gemini TTS only (needs GEMINI_API_KEY; the free tier has a daily request cap).
  microsoft  Microsoft Read Aloud voices Farid / Dilara through edge-tts only (unofficial, no key).
  auto       (default) Gemini chunk by chunk; the moment Gemini can't go on (daily limit, outage,
             time budget) the REMAINING chunks are read by Farid / Dilara. One MP3 either way.

Script format (same rules as the dashboard's parsePodcastScript):
    آقا: ...   / Man: ... / Host: ...   -> male voice
    خانم: ...  / Woman: ... / Guest: ... -> female voice
    A line without a label continues the previous speaker's turn.
    If the script has no labels at all, the whole text is read with the male voice.

Two ways to run
  * JOB_ID set (dashboard -> Apps Script -> this workflow): fetch the script from Apps Script,
    build the MP3, upload it straight into Drive (resumable upload, no size limit).
  * JOB_ID empty (manual "Run workflow" test on GitHub): read SCRIPT_TEXT ("\\n" = new line) and
    write the MP3(s) into out/.

The repo is public, so nothing printed here may contain the script text, URLs or secrets.
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
import wave

import requests

# ---------------- settings ----------------
ENGINE = (os.environ.get("TTS_ENGINE") or "auto").strip().lower()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = [m.strip() for m in (os.environ.get("GEMINI_MODEL") or
                 "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts").split(",") if m.strip()]
GEMINI_MALE = os.environ.get("GEMINI_MALE_VOICE") or "Puck"      # "Upbeat" — chosen by voice test
GEMINI_FEMALE = os.environ.get("GEMINI_FEMALE_VOICE") or "Kore"  # "Firm"
GEMINI_STYLE = os.environ.get("GEMINI_STYLE") or (
    "relaxed, conversational podcast between two friends; natural, lively intonation; moderate pace")
# The man gets his own delivery note (the woman's, above, is the one that already sounds right).
GEMINI_MALE_STYLE = os.environ.get("GEMINI_MALE_STYLE") or (
    "conversational podcast between two friends; energetic and engaged, with clear enthusiasm; "
    "bright, confident voice with expressive, lively intonation that rises and falls naturally; "
    "natural conversational pace, neither rushed nor sleepy; match the female speaker's energy")
STYLE = {"male": GEMINI_MALE_STYLE, "female": GEMINI_STYLE}
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS") or "6000")
# Truncation guard: Persian speech runs ~12-16 characters per second. A chunk whose audio is
# much shorter than chars/25 seconds was cut off by the model -> split it in two and retry.
MIN_CHARS_FOR_CHECK = 600
CHARS_PER_SEC_MAX = 25.0
MAX_SPLIT_DEPTH = 3
# After this many minutes Gemini hands the rest to Microsoft, so the dashboard's 45-min wait holds.
GEMINI_TIME_BUDGET = float(os.environ.get("GEMINI_TIME_BUDGET_MIN") or "30") * 60
API = "https://generativelanguage.googleapis.com/v1beta"

MS_MALE = os.environ.get("MALE_VOICE") or "fa-IR-FaridNeural"
MS_FEMALE = os.environ.get("FEMALE_VOICE") or "fa-IR-DilaraNeural"
MS_RATE = os.environ.get("TTS_RATE") or "+0%"
MS_CONCURRENCY = 3

PAUSE_SEC = float(os.environ.get("PAUSE_SEC") or "0.45")
SAMPLE_RATE = 24000
BITRATE_K = 64
# Evens out loudness over ~2-second windows, so a quieter speaker (Gemini's man) comes up to the
# other's level: 200 ms frames, 9-frame smoothing, peaks to 90 %, at most 8x (18 dB) boost.
LEVEL_FILTER = os.environ.get("LEVEL_FILTER") or "dynaudnorm=f=200:g=9:p=0.9:m=8"
UPLOAD_CHUNK = 8 * 1024 * 1024  # must be a multiple of 256 KB

GAS_URL = os.environ.get("GAS_URL", "").strip()
SECRET = os.environ.get("TTS_SECRET", "").strip()
JOB_ID = os.environ.get("JOB_ID", "").strip()

LABEL_RE = re.compile(r"^(آقا|خانم|man|woman|host|guest)\s*:\s*(.*)$", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"(?<=[.!?؟!…:؛])\s+")

_SECRET_URLS = []  # upload session URLs etc. — hidden from every message


class QuotaExhausted(RuntimeError):
    """Gemini's daily free-tier cap is used up — no point retrying today."""


def redact(msg):
    """Never let the Apps Script URL, upload URLs or the secrets reach the (public) run page."""
    msg = str(msg)
    for s in (GAS_URL, SECRET, GEMINI_KEY, *_SECRET_URLS):
        if s:
            msg = msg.replace(s, "<hidden>")
    msg = re.sub(r"upload_id=[A-Za-z0-9_-]+", "upload_id=<hidden>", msg)
    return re.sub(r"/macros/s/[A-Za-z0-9_-]+", "/macros/s/<hidden>", msg)


def annotate(level, msg):
    """Emit a GitHub Actions annotation (notice/warning/error). Annotations show on the run's
    summary page even when the full log can't be opened, so every key step reports through them."""
    msg = redact(msg).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level}::{msg}", flush=True)


def fmt_time(sec):
    sec = int(round(sec))
    h, rest = divmod(sec, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------- progress reporting (to the dashboard, through Apps Script) ----------------
_last_progress = [0.0, ""]


def progress(message, force=False):
    print(message, flush=True)
    if not JOB_ID or not GAS_URL:
        return
    now = time.time()
    if not force and message == _last_progress[1]:
        return
    if not force and now - _last_progress[0] < 15:
        return
    _last_progress[:] = [now, message]
    try:
        gas_post({"action": "ttsprogress", "job": JOB_ID, "secret": SECRET, "message": message[:200]})
    except Exception as e:  # progress is best-effort
        print(redact(f"(progress report failed: {e})"), flush=True)


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


def split_text(text, max_chars):
    """Split one long turn at sentence ends (or spaces) into pieces of at most ~max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in SENTENCE_END_RE.split(text):
        while len(sent) > max_chars:  # a "sentence" with no punctuation at all: cut at a space
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        parts.append(cur)
    return [p for p in parts if p]


def chunk_turns(segments, max_chars):
    """Group consecutive turns into chunks of at most ~max_chars; over-long turns are split first."""
    flat = [(spk, piece) for spk, t in segments for piece in split_text(t, max_chars)]
    chunks, cur, size = [], [], 0
    for spk, t in flat:
        if cur and size + len(t) > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append((spk, t))
        size += len(t)
    if cur:
        chunks.append(cur)
    return chunks


def halve(turns):
    """Split a chunk into two halves of roughly equal characters (splitting a turn if needed)."""
    total = sum(len(t) for _, t in turns)
    if len(turns) == 1:
        spk, t = turns[0]
        pieces = split_text(t, max(200, len(t) // 2 + 50))
        if len(pieces) < 2:
            mid = t.rfind(" ", 0, len(t) // 2 + 1)
            mid = mid if mid > 0 else len(t) // 2
            pieces = [t[:mid].strip(), t[mid:].strip()]
        first = pieces[:len(pieces) // 2] or pieces[:1]
        second = pieces[len(first):]
        return [(spk, " ".join(first))], [(spk, " ".join(second))]
    acc = 0
    for i, (_, t) in enumerate(turns):
        acc += len(t)
        if acc >= total / 2:
            i = max(1, min(i + (1 if acc - len(t) / 2 < total / 2 else 0), len(turns) - 1))
            return turns[:i], turns[i:]
    return turns[:1], turns[1:]


# ---------------- audio helpers ----------------
def run(cmd):
    subprocess.run(cmd, check=True)


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def normalize(src, dst, fmt_args=()):
    """Any audio -> 24 kHz mono 16-bit WAV, so every piece joins cleanly and can be measured."""
    run(["ffmpeg", "-y", "-loglevel", "error", *fmt_args, "-i", src, "-map_metadata", "-1",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-fflags", "+bitexact", dst])


def audio_bytes_to_wav(audio_bytes, path, rate=SAMPLE_RATE):
    """Gemini returns WAV (RIFF) for normal requests, raw 16-bit PCM in some cases."""
    raw = path + (".src.wav" if audio_bytes[:4] == b"RIFF" else ".pcm")
    with open(raw, "wb") as f:
        f.write(audio_bytes)
    fmt = () if raw.endswith(".wav") else ("-f", "s16le", "-ar", str(rate), "-ac", "1")
    normalize(raw, path, fmt)
    os.remove(raw)


def join_to_mp3(paths, out_path, workdir):
    """Concatenate normalised WAV pieces with a short pause between them; encode one CBR MP3
    without ID3/Xing headers, so every frame has the same size (the dashboard streams by frames)."""
    silence = os.path.join(workdir, "silence.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
         "-t", str(PAUSE_SEC), "-c:a", "pcm_s16le", "-fflags", "+bitexact", silence])
    listfile = os.path.join(workdir, "list.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for i, p in enumerate(paths):
            if i:
                f.write(f"file '{silence}'\n")
            f.write(f"file '{p}'\n")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listfile,
         "-af", LEVEL_FILTER,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", f"{BITRATE_K}k",
         "-write_xing", "0", "-id3v2_version", "0", "-map_metadata", "-1", out_path])


_BITRATES_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]  # MPEG-2/2.5 Layer III
_BITRATES_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_layout(data):
    """Where the audio frames start, and each frame's size/duration (CBR file)."""
    pos = 0
    if data[:3] == b"ID3":
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        pos = 10 + size
    while pos + 4 <= len(data):
        b1, b2, b3 = data[pos + 1], data[pos + 2], data[pos + 3]
        if data[pos] == 0xFF and (b1 & 0xE0) == 0xE0:
            ver = (b1 >> 3) & 3
            if ver != 1 and ((b1 >> 1) & 3) == 1:  # Layer III
                br_i, sr_i = b2 >> 4, (b2 >> 2) & 3
                if 0 < br_i < 15 and sr_i < 3:
                    rate = _RATES[ver][sr_i]
                    kbps = (_BITRATES_V1 if ver == 3 else _BITRATES_V2)[br_i]
                    samples = 1152 if ver == 3 else 576
                    frame_bytes = samples // 8 * kbps * 1000 // rate
                    frame = data[pos:pos + frame_bytes]
                    if b"Xing" in frame[:64] or b"Info" in frame[:64]:
                        pos += frame_bytes  # skip the tag frame
                        continue
                    return {"audioStart": pos, "frameBytes": frame_bytes,
                            "frameSec": samples / rate, "sampleRate": rate, "bitrate": kbps}
        pos += 1
    raise RuntimeError("could not find MP3 frames in the output")


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


def _gemini_bodies(model, turns, male_voice=None):
    male_voice = male_voice or GEMINI_MALE
    names = {"male": "Man", "female": "Woman"}
    interactions = {
        "model": model,
        "input": [{"type": "user_input", "content": [
            {"type": "text", "text": t,
             "annotations": [{"type": "speech_metadata", "speaker": names[s], "style": STYLE[s]}]}
            for s, t in turns]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": {"mode": "conversational", "speakers": [
            {"speaker": "Man", "voice": male_voice}, {"speaker": "Woman", "voice": GEMINI_FEMALE}]}},
    }
    legacy = {
        "contents": [{"role": "user", "parts": [
            {"text": t, "speech_metadata": {"speaker": names[s], "style": STYLE[s]}} for s, t in turns]}],
        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"multiSpeakerVoiceConfig": {
            "speakerVoiceConfigs": [
                {"speaker": "Man", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": male_voice}}},
                {"speaker": "Woman", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_FEMALE}}}]}}},
    }
    return [("interactions", f"{API}/interactions", interactions),
            ("generateContent", f"{API}/models/{model}:generateContent", legacy)]


def _retry_delay(resp, attempt):
    m = re.search(r'"retryDelay"\s*:\s*"(\d+)', resp.text)
    return min(int(m.group(1)) + 2, 90) if m else 15 * (attempt + 1)


_LAST_MODEL = [""]  # model that answered the most recent request (reported per chunk)
_WORKING = None  # (model, shape) that last succeeded — tried first so later chunks don't waste requests


def gemini_request(turns, male_voice=None):
    """One request (up to 2 speakers) -> (audio bytes, sample rate). Tries each model and both API shapes."""
    global _WORKING
    headers = {"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}
    errors = []
    candidates = [(model, shape, url, body) for model in GEMINI_MODELS for shape, url, body in _gemini_bodies(model, turns, male_voice)]
    candidates.sort(key=lambda c: (c[0], c[1]) != _WORKING)  # stable: known-good first, rest keep their order
    for model, shape, url, body in candidates:
        for attempt in range(5):
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=300)
            if resp.status_code == 200:
                data, mime = find_audio(resp.json())
                if data:
                    _WORKING = (model, shape)
                    _LAST_MODEL[0] = model
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


def gemini_pieces(turns, workdir, tag, depth=0, male_voice=None):
    """Read one chunk with Gemini -> list of (normalised WAV path, model). If the audio came back
    suspiciously short (the model stopped early), split the chunk in two and read each half."""
    audio, rate = gemini_request(turns, male_voice)
    model = _LAST_MODEL[0]
    path = os.path.join(workdir, f"g_{tag}.wav")
    audio_bytes_to_wav(audio, path, rate)
    chars = sum(len(t) for _, t in turns)
    secs = wav_seconds(path)
    if chars > MIN_CHARS_FOR_CHECK and secs < chars / CHARS_PER_SEC_MAX:
        if depth >= MAX_SPLIT_DEPTH:
            annotate("warning", f"Gemini audio still looks cut short after {depth} split(s) "
                                f"({chars} chars -> {secs:.0f}s); keeping it")
            return [(path, model)]
        annotate("warning", f"Gemini audio looks cut short ({chars} chars -> {secs:.0f}s); splitting and retrying")
        os.remove(path)
        a, b = halve(turns)
        return (gemini_pieces(a, workdir, tag + "a", depth + 1, male_voice) +
                gemini_pieces(b, workdir, tag + "b", depth + 1, male_voice))
    return [(path, model)]


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


async def _ms_all(turns, workdir, tag):
    sem = asyncio.Semaphore(MS_CONCURRENCY)
    mp3s = [os.path.join(workdir, f"m_{tag}_{i:03d}.mp3") for i in range(len(turns))]

    async def one(i, spk, text):
        async with sem:
            await _ms_one(text, MS_MALE if spk == "male" else MS_FEMALE, mp3s[i])

    await asyncio.gather(*(one(i, s, t) for i, (s, t) in enumerate(turns)))
    return mp3s


def microsoft_pieces(turns, workdir, tag):
    """Read one chunk with Farid / Dilara (one request per turn) -> list of normalised WAV paths."""
    wavs = []
    for mp3 in asyncio.run(_ms_all(turns, workdir, tag)):
        wav = mp3[:-4] + ".wav"
        normalize(mp3, wav)
        os.remove(mp3)
        wavs.append(wav)
    return wavs


# ---------------- orchestration ----------------
ENGINE_LABEL = {"gemini": f"Gemini ({GEMINI_MALE} / {GEMINI_FEMALE})", "microsoft": "Microsoft Farid / Dilara"}


VOICE_TEST_RE = re.compile(r"^\s*VOICES\s*:\s*(.+)$", re.IGNORECASE)


def split_voice_test(text):
    """A first line "VOICES: Puck, Achird, ..." asks for a male-voice audition instead of a podcast."""
    lines = text.strip().splitlines()
    m = VOICE_TEST_RE.match(lines[0]) if lines else None
    if not m:
        return [], text
    voices = []
    for v in re.split(r"[,،\s]+", m.group(1)):
        v = v.strip().capitalize()
        if re.fullmatch(r"[A-Z][a-z]{2,20}", v) and v not in voices:
            voices.append(v)
    return voices[:6], "\n".join(lines[1:])


def build_voice_test(voices, segments):
    """The same sample read once per male voice (the woman stays the same), one after another."""
    sample = chunk_turns(segments, CHUNK_CHARS)[0]
    if len(sample) < len(segments):
        annotate("notice", f"Voice test uses only the first {len(sample)} turn(s) of the sample")
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    with tempfile.TemporaryDirectory() as workdir:
        gap = os.path.join(workdir, "gap.wav")
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
             "-t", "1.5", "-c:a", "pcm_s16le", "-fflags", "+bitexact", gap])
        paths, marks, t = [], [], 0.0
        for i, v in enumerate(voices):
            progress(f"Voice test: {v} ({i + 1}/{len(voices)})", force=True)
            try:
                got = [p for p, _ in gemini_pieces(sample, workdir, f"v{i}", male_voice=v)]
            except QuotaExhausted:
                marks += [f"{x} — not tested (Gemini daily limit)" for x in voices[i:]]
                annotate("warning", "Gemini daily limit reached during the voice test")
                break
            except Exception as e:
                marks.append(f"{v} — failed")
                annotate("warning", f"Voice {v} failed: {str(e)[:300]}")
                continue
            if paths:
                paths.append(gap)
                t += wav_seconds(gap) + PAUSE_SEC
            marks.append(f"{v} {fmt_time(t)}")
            for p in got:
                paths.append(p)
                t += wav_seconds(p) + PAUSE_SEC
        if not paths:
            raise RuntimeError("no voice could be tested — " + "; ".join(marks))
        progress("Encoding the MP3", force=True)
        out = os.path.join(workdir, "podcast.mp3")
        join_to_mp3(paths, out, workdir)
        with open(out, "rb") as f:
            data = f.read()
    layout = mp3_layout(data)
    duration = (len(data) - layout["audioStart"]) // layout["frameBytes"] * layout["frameSec"]
    note = f"Voice test (man; woman = {GEMINI_FEMALE}): " + " · ".join(marks)
    annotate("notice", note)
    return data, {"duration": round(duration, 2), "engine": "gemini", "note": note, "size": len(data), **layout}


def build_mp3(text, engine=ENGINE):
    """Returns (mp3_bytes, meta). meta: duration, engine, note, audioStart, frameBytes, frameSec, ..."""
    voices, text = split_voice_test(text)
    segments = parse_script(text)
    if not segments:
        raise RuntimeError("the script is empty")
    if voices:
        return build_voice_test(voices, segments)
    total_chars = sum(len(t) for _, t in segments)
    chunks = chunk_turns(segments, CHUNK_CHARS)
    print(f"{len(segments)} turn(s), {total_chars} characters, {len(chunks)} chunk(s) of up to {CHUNK_CHARS}")

    use_gemini = engine in ("auto", "gemini")
    reason = ""
    if use_gemini and not GEMINI_KEY:
        if engine == "gemini":
            raise RuntimeError("GEMINI_API_KEY is not set")
        use_gemini, reason = False, "no Gemini key"

    started = time.time()
    with tempfile.TemporaryDirectory() as workdir:
        pieces = []            # (engine, wav path, chunk no., model) in playback order
        for i, turns in enumerate(chunks):
            tag = f"{i:03d}"
            if use_gemini and engine == "auto" and time.time() - started > GEMINI_TIME_BUDGET:
                use_gemini, reason = False, "Gemini time budget"
                annotate("warning", f"Gemini time budget used up at chunk {i + 1}/{len(chunks)}; Microsoft reads the rest")
            if use_gemini:
                progress(f"Gemini: chunk {i + 1}/{len(chunks)}", force=(i == 0))
                try:
                    pieces += [("gemini", p, i + 1, m) for p, m in gemini_pieces(turns, workdir, tag)]
                    continue
                except Exception as e:
                    if engine == "gemini":
                        raise
                    reason = "Gemini daily limit" if isinstance(e, QuotaExhausted) else "Gemini error"
                    annotate("warning", f"Gemini stopped at chunk {i + 1}/{len(chunks)} ({str(e)[:300]}); "
                                        f"Microsoft reads the remaining {len(chunks) - i} chunk(s)")
                    use_gemini = False
            progress(f"Microsoft Farid / Dilara: chunk {i + 1}/{len(chunks)}", force=True)
            pieces += [("microsoft", p, i + 1, "microsoft") for p in microsoft_pieces(turns, workdir, tag)]

        progress("Encoding the MP3", force=True)
        out = os.path.join(workdir, "podcast.mp3")
        join_to_mp3([p[1] for p in pieces], out, workdir)
        with open(out, "rb") as f:
            data = f.read()

        # where did the engine change (for the note under the player)?
        t, switch_at, starts = 0.0, None, []
        for k, (eng, p, chunk, model) in enumerate(pieces):
            if k and eng != pieces[k - 1][0] and switch_at is None:
                switch_at = t
            if not k or chunk != pieces[k - 1][2] or model != pieces[k - 1][3]:
                starts.append((chunk, t, model))
            t += wav_seconds(p) + PAUSE_SEC
        engines_used = {p[0] for p in pieces}
    # where each chunk starts and which model read it — to match a voice change heard in the MP3
    short = lambda m: m.replace("gemini-", "")
    annotate("notice", "Chunk start times: " + " · ".join(f"#{c} {fmt_time(t0)} {short(m)}" for c, t0, m in starts))
    gem_models = sorted({p[3] for p in pieces if p[0] == "gemini"})
    if len(gem_models) > 1:
        annotate("warning", "Gemini used more than one model (" + ", ".join(gem_models) +
                            ") — the voices may sound different where the model changes")

    layout = mp3_layout(data)
    frames = (len(data) - layout["audioStart"]) // layout["frameBytes"]
    duration = frames * layout["frameSec"]
    if engines_used == {"gemini", "microsoft"}:
        used = "mixed"
        note = f"Gemini until {fmt_time(switch_at)}, then Microsoft Farid / Dilara ({reason})"
    else:
        used = engines_used.pop()
        note = "Voices: " + ENGINE_LABEL[used] + (f" ({reason})" if used == "microsoft" and reason else "")
    meta = {"duration": round(duration, 2), "engine": used, "note": note, "size": len(data), **layout}
    msg = f"MP3 built: {len(data) / 1e6:.2f} MB, {fmt_time(duration)}, {note}"
    print(msg)
    annotate("notice", msg)
    return data, meta


# ---------------- Apps Script / Drive ----------------
def _gas_json(r, what):
    if r.status_code >= 400:
        raise RuntimeError(f"Apps Script {what}: HTTP {r.status_code} {r.text[:300]}")
    try:
        return r.json()
    except ValueError:
        # Apps Script answers script errors with an HTML page, not JSON — surface its text
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text)).strip()
        raise RuntimeError(f"Apps Script {what}: not JSON — {text[:300]}")


def gas_get(params):
    return _gas_json(requests.get(GAS_URL, params=params, timeout=90), params.get("action", "GET"))


def gas_post(data):
    # Apps Script answers a POST with a redirect; the request is already processed by then.
    res = _gas_json(requests.post(GAS_URL, data=data, timeout=300), data.get("action", "POST"))
    if res.get("status") == "error":
        raise RuntimeError(f"Apps Script {data.get('action')}: {res.get('message', 'error')}")
    return res


def upload_to_drive(upload_url, data):
    """Resumable upload into the session Apps Script opened (the URL itself is the permission).
    Returns the new file's id."""
    _SECRET_URLS.append(upload_url)
    total, pos = len(data), 0
    while pos < total:
        end = min(pos + UPLOAD_CHUNK, total)
        for attempt in range(5):
            try:
                r = requests.put(upload_url, data=data[pos:end], timeout=300, headers={
                    "Content-Length": str(end - pos), "Content-Range": f"bytes {pos}-{end - 1}/{total}"})
            except requests.RequestException as e:
                r, err = None, e
            else:
                err = None
            if r is not None and r.status_code in (200, 201):
                return r.json()["id"]
            if r is not None and r.status_code == 308:
                rng = r.headers.get("Range")  # e.g. "bytes=0-8388607": what Drive actually has
                pos = int(rng.split("-")[1]) + 1 if rng else 0
                break
            if r is not None and r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"Drive upload: HTTP {r.status_code} {r.text[:200]}")
            time.sleep(5 * (attempt + 1))
            # ask Drive how far it got before retrying
            q = requests.put(upload_url, timeout=60, headers={"Content-Length": "0", "Content-Range": f"bytes */{total}"})
            if q.status_code in (200, 201):
                return q.json()["id"]
            if q.status_code == 308:
                rng = q.headers.get("Range")
                pos = int(rng.split("-")[1]) + 1 if rng else 0
                end = min(pos + UPLOAD_CHUNK, total)
        else:
            raise RuntimeError(f"Drive upload kept failing: {err or (r.status_code if r is not None else '')}")
        progress(f"Uploading to Drive: {pos * 100 // total}%")
    raise RuntimeError("Drive upload ended without a file id")


def main():
    if JOB_ID:
        if not GAS_URL or not SECRET:
            annotate("error", "Repository secrets GAS_URL and TTS_SECRET must both be set.")
            sys.exit(1)
        stage = "fetching the script from Apps Script"
        try:
            info = gas_get({"action": "ttsjob", "job": JOB_ID, "secret": SECRET})
            if info.get("status") != "ok":
                raise RuntimeError(info.get("message", info.get("status")))
            annotate("notice", f"Script received: {len(info.get('text', ''))} characters")
            stage = "building the audio"
            data, meta = build_mp3(info["text"])
            stage = "opening the Drive upload"
            progress("Uploading to Drive: 0%", force=True)
            up = gas_post({"action": "ttsupload", "job": JOB_ID, "secret": SECRET, "size": str(len(data))})
            stage = "uploading the MP3 to Drive"
            file_id = upload_to_drive(up["uploadUrl"], data)
            stage = "finishing up in Apps Script"
            res = gas_post({"action": "ttsdone", "job": JOB_ID, "secret": SECRET, "fileId": file_id,
                            "meta": json.dumps(meta)})
            # the repo is public, so its run page is too: report the outcome only, never the script text
            annotate("notice", f"Saved to Drive: {res.get('filename', 'MP3')} — {meta['note']}")
        except Exception as e:
            msg = f"Failed while {stage}: {e}"
            annotate("error", msg[:900])
            try:  # tell the dashboard right away instead of letting it wait
                gas_post({"action": "ttserror", "job": JOB_ID, "secret": SECRET, "message": redact(msg)[:500]})
            except Exception as e2:
                annotate("error", f"Could not report the failure to Apps Script either: {e2}"[:900])
            sys.exit(1)
    else:
        text = (os.environ.get("SCRIPT_TEXT") or "").replace("\\n", "\n")
        os.makedirs("out", exist_ok=True)
        engines = ["gemini", "microsoft"] if ENGINE == "auto" else [ENGINE]
        made = 0
        for eng in engines:
            try:
                data, meta = build_mp3(text, eng)
                with open(f"out/podcast_{eng}.mp3", "wb") as f:
                    f.write(data)
                made += 1
            except Exception as e:
                annotate("warning", f"Test with {eng} failed: {e}"[:900])
        if not made:
            sys.exit("No engine produced audio — see the errors above.")
        print("Test MP3(s) written to out/ — download them from the run's Artifacts.")


if __name__ == "__main__":
    main()
