import os
import json
from pydub import AudioSegment
from langdetect import detect, DetectorFactory
from typing import List, Dict, Tuple
import warnings
import numpy

DetectorFactory.seed = 0
warnings.filterwarnings("ignore")

# ---------------- LANGUAGE DETECTION ----------------
def detect_language(text: str) -> str:
    lang = detect(text)
    return "en" if lang == "en" else "hi"

# ---------------- PHONEME EXTRACTION ----------------
def get_phonemes_en(text: str) -> List[str]:
    try:
        from eng_to_ipa import convert
        ipa = convert(text, stress_marks=False, punctuation=False)
        ipa = ipa.replace("ˈ", "").replace("ˌ", "").replace("ɹ", "r")
        return [p for p in ipa if p.isalpha()]
    except:
        return list(text)

def get_phonemes_hi_gu(text: str) -> List[str]:
    clusters = [
        "kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh",
        "aa", "ii", "uu", "ai", "au", "sh", "ng", "ñ", "ṇ", "ṛ",
        "a", "e", "i", "o", "u", "k", "g", "c", "j", "ṭ", "ḍ", "t", "d",
        "n", "p", "b", "m", "y", "r", "l", "v", "s", "h"
    ]
    text = text.lower()
    phonemes = []
    i = 0
    while i < len(text):
        match = None
        for c in sorted(clusters, key=len, reverse=True):
            if text[i:i+len(c)] == c:
                match = c
                break
        if match:
            phonemes.append(match)
            i += len(match)
        else:
            if text[i].isalpha():
                phonemes.append(text[i])
            i += 1
    return phonemes

# ---------------- SILENCE DETECTION ----------------
def detect_silences(audio: AudioSegment, silence_thresh=-40.0, min_silence_len=200) -> List[Tuple[int, int]]:
    silence_ranges = []
    window_ms = 50
    silence_start = None
    for i in range(0, len(audio), window_ms):
        segment = audio[i:i + window_ms]
        if segment.dBFS < silence_thresh:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None and (i - silence_start) >= min_silence_len:
                silence_ranges.append((silence_start, i))
            silence_start = None
    if silence_start is not None and (len(audio) - silence_start) >= min_silence_len:
        silence_ranges.append((silence_start, len(audio)))
    return silence_ranges

# ---------------- PHONEME TO VISEME ----------------
phoneme_to_viseme = {
    "A": ["aa", "ae", "ah", "ao", "aw", "ay", "a", "e", "i", "o", "u", "uh", "uw"],
    "B": ["b", "p", "m", "em", "bm"],
    "C": ["ch", "jh", "sh", "zh"],
    "D": ["d", "dh", "t", "th", "n", "nd", "nt"],
    "E": ["eh", "ey", "iy", "y", "ee", "ih"],
    "F": ["f", "v", "w", "wh"],
    "G": ["g", "gh", "k", "kh", "ng"],
    "H": ["h", "hh", "l", "r", "s", "z", "ll", "rr"],
    "X": [" ", "sil", "sp", "pause"]
}

def map_to_viseme(phoneme: str) -> str:
    for viseme, phonemes in phoneme_to_viseme.items():
        if phoneme in phonemes:
            return viseme
    return "X"

# ---------------- CREATE VISEME CUES ----------------
def create_viseme_cues(phonemes: List[str], duration_ms: int, silences: List[Tuple[int, int]]) -> List[Dict]:
    cues = []
    non_silent_ranges = []
    last = 0
    for s_start, s_end in silences:
        if last < s_start:
            non_silent_ranges.append((last, s_start))
        last = s_end
    if last < duration_ms:
        non_silent_ranges.append((last, duration_ms))

    total_active_duration = sum(e - s for s, e in non_silent_ranges)
    if total_active_duration <= 0 or not phonemes:
        return []

    phoneme_duration = total_active_duration / len(phonemes)
    current_phoneme_index = 0

    for segment in sorted(non_silent_ranges + silences, key=lambda x: x[0]):
        seg_start, seg_end = segment
        seg_duration = seg_end - seg_start

        if (seg_start, seg_end) in silences:
            cues.append({
                "value": "X",
                "start": round(seg_start / 1000, 2),
                "end": round(seg_end / 1000, 2)
            })
        else:
            t = seg_start
            while t + phoneme_duration <= seg_end and current_phoneme_index < len(phonemes):
                ph = phonemes[current_phoneme_index]
                viseme = map_to_viseme(ph)
                cues.append({
                    "value": viseme,
                    "start": round(t / 1000, 2),
                    "end": round((t + phoneme_duration) / 1000, 2)
                })
                t += phoneme_duration
                current_phoneme_index += 1

    return cues

# ---------------- MERGE REPEATED VISEMES ----------------
def merge_repeated_visemes(cues: List[Dict]) -> List[Dict]:
    if not cues:
        return []

    merged = [cues[0]]
    for cue in cues[1:]:
        last = merged[-1]
        if cue["value"] == last["value"]:
            last["end"] = cue["end"]
        else:
            merged.append(cue)
    return merged

# ---------------- MAIN PROCESS ----------------
def process(text: str, audio_path: str, output_path: str = "visemes.json"):
    print("🌐 Detecting language...")
    lang = detect_language(text)
    print(f"🈯 Language: {lang}")

    print("🔤 Converting text to phonemes...")
    phonemes = get_phonemes_en(text) if lang == "en" else get_phonemes_hi_gu(text)
    print(f"🔡 Phonemes: {phonemes}")

    print("🎧 Loading audio...")
    audio = AudioSegment.from_file(audio_path).set_channels(1).set_sample_width(2).set_frame_rate(44100)
    duration_ms = len(audio)

    print("🔇 Detecting silences...")
    silences = detect_silences(audio)
    print(f"⏱️ Duration: {duration_ms}ms | Silences: {silences}")

    print("🎭 Creating viseme timeline...")
    cues = create_viseme_cues(phonemes, duration_ms, silences)

    print("🪄 Merging repeated visemes...")
    cues = merge_repeated_visemes(cues)

    print(f"💾 Saving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(cues, f, indent=2)

    print("✅ Done! Total visemes:", len(cues))

# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    process(
        text="I was so happy when I saw you waiting for me at the station. My heart literally jumped, but then I noticed the message on my phone. And I just froze. Why didn't you tell me earlier? I felt so angry like everything was falling apart, and honestly, a part of me was scared. Scared that I'd lose you. Still, right now, just be able to talk to you. It makes me feel calm again. I just hope things can go back to how they were.",
        audio_path="Harshil.wav"
    )