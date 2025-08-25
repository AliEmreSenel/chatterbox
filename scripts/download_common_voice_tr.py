import argparse
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset


def main(out_dir: str, split: str = "train", limit: int = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds_name_candidates = [
        ("mozilla-foundation/common_voice_13_0"),
    ]
    ds = load_dataset("mozilla-foundation/common_voice_13_0", "tr", split=split)
    meta = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        audio = item.get("audio") or item.get("path")
        text = (
            item.get("sentence")
            or item.get("text")
            or item.get("transcription")
            or item.get("sentence")
        )
        if audio is None and item.get("audio") is None:
            continue
        if isinstance(audio, dict) and "array" in audio:
            wav = audio["array"]
            sr = audio.get("sampling_rate", 16000)
        else:
            path = audio if isinstance(audio, str) else audio.get("path")
            wav, sr = librosa.load(path, sr=16000)
        wav_path = out / f"cv_tr_{i:06d}.wav"
        sf.write(str(wav_path), wav, 16000)
        meta.append(f"{wav_path}\t{text}\n")
    with open(out / "metadata.tsv", "w", encoding="utf-8") as f:
        f.writelines(meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/common_voice_tr")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.out_dir, args.split, args.limit)
