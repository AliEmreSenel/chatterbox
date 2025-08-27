"""Download and extract Mozilla Common Voice v22 tarball (official release) for Turkish.

Usage:
  - If you already have the cv-corpus-22.0-...tar.gz locally: --tarball /path/to/tar.gz
  - Or provide a direct URL to the tar.gz: --url <download_url>
  - The script will extract the Turkish files, convert clips to 16k mono WAV using ffmpeg, and write metadata.tsv

Caveats: the tarball is large (many GB). ffmpeg must be installed and on PATH.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

try:
    import requests
except Exception:
    requests = None


def run_ffmpeg_convert(in_path: Path, out_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def find_tr_dir(extracted_root: Path):
    # find a directory that endswith '/tr' under extracted_root
    for p in extracted_root.rglob("tr"):
        if p.is_dir():
            return p
    return None


def main(
    out_dir: str,
    tarball: str = None,
    url: str = None,
    split: str = "train",
    limit: int = None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if tarball is None and url is None:
        print("Provide --tarball or --url (direct link to cv-corpus-22 tar.gz)")
        sys.exit(1)

    tmp = tempfile.mkdtemp(prefix="cv22_")
    try:
        local_tar = Path(tarball) if tarball else Path(tmp) / "cv.tar.gz"
        if url:
            if requests is None:
                print("requests is required to download from URL; pip install requests")
                sys.exit(1)
            print(f"Downloading {url} ...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_tar, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        print("Extracting tarball (this may take a while) ...")
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=tmp)

        extracted_root = Path(tmp)
        tr_dir = find_tr_dir(extracted_root)
        if tr_dir is None:
            # Sometimes structure is cv-corpus-22.0-*/cv-corpus-22.0-*/tr
            # try one level deeper
            for d in extracted_root.iterdir():
                maybe = find_tr_dir(d)
                if maybe:
                    tr_dir = maybe
                    break
        if tr_dir is None:
            print("Could not locate Turkish (tr) directory inside the tarball")
            sys.exit(1)

        # locate clips dir and tsv for split
        clips_dir = tr_dir / "clips"
        tsv_path = tr_dir / f"{split}.tsv"
        if not tsv_path.exists():
            # sometimes files are under "validated" etc; search for any *.tsv in tr_dir
            tsvs = list(tr_dir.glob("*.tsv"))
            if not tsvs:
                print("No TSV files found in tr dir")
                sys.exit(1)
            # prefer train.tsv
            for t in tsvs:
                if "train" in t.name:
                    tsv_path = t
                    break
            else:
                tsv_path = tsvs[0]

        meta_out = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                path = row.get("path")
                sentence = (
                    row.get("sentence") or row.get("sentence") or row.get("text") or ""
                )
                if path is None:
                    continue
                # the path may be just filename inside clips_dir
                src = clips_dir / path if clips_dir.exists() else tr_dir / path
                if not src.exists():
                    # try search
                    cand = list(tr_dir.rglob(path))
                    if cand:
                        src = cand[0]
                    else:
                        print(f"Source audio not found for {path}; skipping")
                        continue
                out_wav = Path(out) / f"cv_tr_{i:06d}.wav"
                ok = run_ffmpeg_convert(src, out_wav)
                if not ok:
                    print(f"ffmpeg failed for {src}; skipping")
                    continue
                meta_out.append(f"{out_wav}\t{sentence}\n")
        with open(Path(out) / "metadata.tsv", "w", encoding="utf-8") as f:
            f.writelines(meta_out)
        print(f"Wrote {len(meta_out)} entries to {Path(out) / 'metadata.tsv'}")
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/mnt/data/AI/common_voice/tr")
    p.add_argument("--tarball", default=None)
    p.add_argument("--url", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.out_dir, args.tarball, args.url, args.split, args.limit)
