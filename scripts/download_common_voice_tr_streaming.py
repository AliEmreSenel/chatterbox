import argparse
import io
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset


def save_audio_from_bytes(bts, out_path):
    # write raw bytes to file then load/resample
    out_path.write_bytes(bts)
    wav, sr = librosa.load(str(out_path), sr=16000)
    return wav, 16000


def main(out_dir: str, split: str = "train", limit: int = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    repo = "mozilla-foundation/common_voice_13_0"
    print(f"Loading streaming dataset {repo} config tr split={split} ...")
    try:
        ds = load_dataset(repo, "tr", split=split, streaming=True, use_auth_token=True)
    except TypeError:
        ds = load_dataset(repo, "tr", split=split, streaming=True)

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
            or ""
        )
        try:
            if isinstance(audio, dict):
                if "array" in audio and audio["array"] is not None:
                    wav = audio["array"]
                    sr = audio.get("sampling_rate", 16000)
                elif "bytes" in audio and audio["bytes"] is not None:
                    # save bytes to a temp file in out dir
                    tmp_path = out / f"cv_tr_{i:06d}.mp3"
                    wav, sr = save_audio_from_bytes(audio["bytes"], tmp_path)
                elif "path" in audio and audio["path"] is not None:
                    # dataset may reference local path in cache; try to load
                    try:
                        wav, sr = librosa.load(audio["path"], sr=16000)
                    except Exception:
                        # fallback: if path is remote, skip
                        print(f"Skipping remote path for item {i}")
                        continue
                else:
                    print(f"Skipping item {i}: unknown audio format")
                    continue
            elif isinstance(audio, str):
                wav, sr = librosa.load(audio, sr=16000)
            else:
                print(f"Skipping item {i}: no audio")
                continue
        except Exception as e:
            print(f"Error loading audio for item {i}: {e}")
            continue

        wav_path = out / f"cv_tr_{i:06d}.wav"
        try:
            sf.write(str(wav_path), wav, 16000)
        except Exception as e:
            print(f"Failed to write wav for item {i}: {e}")
            continue
        meta.append(f"{wav_path}\t{text}\n")

    with open(out / "metadata.tsv", "w", encoding="utf-8") as f:
        f.writelines(meta)
    print(f"Wrote {len(meta)} entries to {out / 'metadata.tsv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/common_voice_tr")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.out_dir, args.split, args.limit)
