#!/usr/bin/env python3
"""Initial data exploration for Common Voice TR dataset.
Usage: python scripts/explore_common_voice_tr.py /mnt/data/AI/common-voice/tr/ --meta dataset.tsv --sample 500 --out report.json
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torchaudio as ta


def read_meta(meta_path: Path):
    items = []
    base = meta_path.resolve().parent
    with open(meta_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            p, text = parts[0], parts[1]
            p_path = Path(p)
            if not p_path.is_absolute():
                cand = base / p
                if cand.exists():
                    wav_path = cand
                elif (Path.cwd() / p).exists():
                    wav_path = Path.cwd() / p
                else:
                    wav_path = None
            else:
                wav_path = p_path if p_path.exists() else None
            items.append(
                {"wav": str(wav_path) if wav_path is not None else None, "text": text}
            )
    return items


def safe_info(path: Path):
    try:
        info = ta.info(str(path))
        frames = info.num_frames
        sr = info.sample_rate
        duration = frames / float(sr)
        return duration, sr
    except Exception:
        return None, None


def explore(root: Path, meta: str, sample: int, out: Path):
    meta_path = Path(meta) if meta else root / "validated.tsv"
    items = read_meta(meta_path)
    total = len(items)
    missing_wavs = [i for i, it in enumerate(items) if it["wav"] is None]
    existing = [it for it in items if it["wav"] is not None]
    report = {"total_records": total, "missing_wav_count": len(missing_wavs)}

    # Text statistics
    texts = [it["text"] for it in items if it["text"] is not None]
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]
    report.update(
        {
            "texts_count": len(texts),
            "char_len": {
                "mean": statistics.mean(char_lens) if char_lens else 0,
                "median": statistics.median(char_lens) if char_lens else 0,
                "min": min(char_lens) if char_lens else 0,
                "max": max(char_lens) if char_lens else 0,
            },
            "word_len": {
                "mean": statistics.mean(word_lens) if word_lens else 0,
                "median": statistics.median(word_lens) if word_lens else 0,
                "min": min(word_lens) if word_lens else 0,
                "max": max(word_lens) if word_lens else 0,
            },
        }
    )

    # Top words
    ctr = Counter()
    for t in texts:
        for w in t.lower().split():
            ctr[w] += 1
    report["top_words"] = ctr.most_common(30)

    # Character set
    chars = Counter()
    for t in texts:
        chars.update(list(t))
    report["top_chars"] = chars.most_common(50)

    # Audio duration stats (sample)
    durations = []
    srs = Counter()
    n = min(sample, len(existing))
    for i in range(n):
        it = existing[i]
        try:
            dur, sr = safe_info(Path(it["wav"]))
            if dur is not None:
                durations.append(dur)
            if sr:
                srs[sr] += 1
        except Exception:
            continue
    if durations:
        report["audio_durations"] = {
            "count": len(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
        }
        report["audio_sample_rates"] = dict(srs)
    else:
        report["audio_durations"] = {"count": 0}

    # Total audio duration across all existing files (may take time)
    total_seconds = 0.0
    found = 0
    durations_all = []
    from tqdm import tqdm as _tqdm

    for it in _tqdm(existing, desc="Scanning all audio durations", unit="file"):
        try:
            dur, _ = safe_info(Path(it["wav"]))
            if dur is not None:
                total_seconds += float(dur)
                durations_all.append(float(dur))
                found += 1
        except Exception:
            continue
    report["total_audio_duration_seconds"] = total_seconds
    report["total_audio_duration_hours"] = total_seconds / 3600.0
    report["total_audio_files_counted"] = found

    # Additional audio duration stats (all files counted)
    if durations_all:
        import numpy as _np

        report["audio_duration_percentiles_seconds"] = {
            "p10": float(_np.percentile(durations_all, 10)),
            "p25": float(_np.percentile(durations_all, 25)),
            "p50": float(_np.percentile(durations_all, 50)),
            "p75": float(_np.percentile(durations_all, 75)),
            "p90": float(_np.percentile(durations_all, 90)),
            "p95": float(_np.percentile(durations_all, 95)),
        }
        # duration buckets
        bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 99999)]
        bin_counts = {}
        for lo, hi in bins:
            bin_counts[f"{lo}-{hi}"] = sum(1 for d in durations_all if lo <= d < hi)
        report["duration_bins_counts"] = bin_counts
    else:
        report["audio_duration_percentiles_seconds"] = {}
        report["duration_bins_counts"] = {}

    # Words-per-audio stats
    words_per_file = [len(t.split()) for t in texts]
    if words_per_file:
        report["words_per_file"] = {
            "mean": float(statistics.mean(words_per_file)),
            "median": float(statistics.median(words_per_file)),
            "min": int(min(words_per_file)),
            "max": int(max(words_per_file)),
        }
    else:
        report["words_per_file"] = {}

    # Speaker distribution (attempt to extract speaker id from filename prefix before first '-')
    speakers = Counter()
    for it in existing:
        wav = it.get("wav")
        if not wav:
            continue
        name = Path(wav).name
        if "-" in name:
            spk = name.split("-", 1)[0]
            speakers[spk] += 1
    report["top_speakers"] = speakers.most_common(50)

    # Additional lexical stats
    all_tokens = []
    for t in texts:
        all_tokens.extend([w.strip() for w in t.lower().split() if w.strip()])
    unique_words = set(all_tokens)
    hapax = [w for w, c in Counter(all_tokens).items() if c == 1]
    report["vocab_size"] = len(unique_words)
    report["hapax_count"] = len(hapax)
    report["lexical_diversity"] = float(len(unique_words)) / max(1, len(all_tokens))
    if all_tokens:
        report["avg_word_length"] = float(statistics.mean([len(w) for w in all_tokens]))
    else:
        report["avg_word_length"] = 0.0

    # Punctuation and character case stats
    import string

    punct_ctr = Counter()
    uppercase_lines = 0
    numeric_token_count = 0
    for t in texts:
        if t.strip() and t.strip() == t.strip().upper():
            uppercase_lines += 1
        for ch in t:
            if ch in string.punctuation:
                punct_ctr[ch] += 1
        for w in t.split():
            if any(c.isdigit() for c in w):
                numeric_token_count += 1
    report["top_punctuation"] = punct_ctr.most_common(30)
    report["percent_uppercase_lines"] = (uppercase_lines / max(1, len(texts))) * 100.0
    report["numeric_token_count"] = numeric_token_count

    # File size distribution for existing audio files (bytes)
    sizes = []
    for it in existing:
        p = it.get("wav")
        if not p:
            continue
        try:
            sz = Path(p).stat().st_size
            sizes.append(sz)
        except Exception:
            continue
    if sizes:
        report["file_size_bytes"] = {
            "count": len(sizes),
            "min": min(sizes),
            "mean": int(statistics.mean(sizes)),
            "median": int(statistics.median(sizes)),
            "max": max(sizes),
        }
    else:
        report["file_size_bytes"] = {"count": 0}

    # Per-speaker duration summary for top speakers
    speaker_durations = {}
    for spk, _ in report["top_speakers"]:
        speaker_durations[spk] = {"total_seconds": 0.0, "count": 0}
    for it in existing:
        wav = it.get("wav")
        if not wav:
            continue
        name = Path(wav).name
        if "-" in name:
            spk = name.split("-", 1)[0]
            if spk in speaker_durations:
                try:
                    dur, _ = safe_info(Path(wav))
                    if dur is not None:
                        speaker_durations[spk]["total_seconds"] += float(dur)
                        speaker_durations[spk]["count"] += 1
                except Exception:
                    pass
    report["top_speakers_durations"] = speaker_durations

    # Save report
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Common Voice TR root dir")
    p.add_argument(
        "--meta", default=None, help="metadata tsv (default: validated.tsv in root)"
    )
    p.add_argument(
        "--sample",
        type=int,
        default=500,
        help="number of audio files to sample for duration stats",
    )
    p.add_argument(
        "--out",
        default="runs/explore_common_voice_tr_report.json",
        help="output JSON report path",
    )
    args = p.parse_args()
    explore(Path(args.root), args.meta, args.sample, Path(args.out))
