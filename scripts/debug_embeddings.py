#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
import torchaudio as ta

from dataset_utils import SimpleMetaDataset, read_meta
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--ckpt_dir", required=False)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    s3tok = S3Tokenizer()
    ds = SimpleMetaDataset(args.meta, s3tok, cache_dir=args.cache_dir)

    s3gen = S3Gen()
    if args.ckpt_dir:
        try:
            state = torch.load(Path(args.ckpt_dir) / "s3gen.safetensors", map_location="cpu")
        except Exception:
            try:
                from safetensors.torch import load_file

                state = load_file(Path(args.ckpt_dir) / "s3gen.safetensors")
            except Exception:
                logger.exception("Failed to load checkpoint; proceeding with random init")
                state = None
        if state is not None:
            try:
                s3gen.load_state_dict(state, strict=False)
            except Exception:
                logger.exception("Failed to load state into model; continuing")

    n = min(args.n, len(ds))
    zero_count = 0
    total = 0
    for idx in range(n):
        item = ds[idx]
        wav_path = item.get("wav")
        cache_used = item.get("cache_path") is not None
        if wav_path is None:
            waveform = torch.zeros(16000)
            sr = 16000
        else:
            try:
                waveform, sr = ta.load(wav_path)
            except Exception:
                waveform = torch.zeros(1, 16000)
                sr = 16000
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        if sr != 16000:
            waveform = ta.transforms.Resample(sr, 16000)(waveform.unsqueeze(0)).squeeze(0)
        wav16 = waveform

        # tokens
        try:
            toks, lens = s3tok([wav16])
            tok = toks[0]
            tok_len = int(lens[0])
        except Exception:
            tok = torch.zeros(1, dtype=torch.long)
            tok_len = 0

        # speaker embedding
        try:
            emb = s3gen.speaker_encoder.inference(wav16.unsqueeze(0))
        except Exception:
            emb = None

        if emb is None:
            logger.info("idx %d: emb=None tok_len=%s wav=%s cache=%s", idx, tok_len, wav_path, cache_used)
            zero_count += 1
        else:
            # emb may be (1, D) or (D,)
            if emb.dim() == 2 and emb.size(0) == 1:
                emb_item = emb.squeeze(0)
            else:
                emb_item = emb.view(-1)
            s = float(emb_item.abs().sum().item())
            if s == 0.0:
                logger.info("idx %d: zero-embedding tok_len=%s wav=%s cache=%s min=%s max=%s mean=%s",
                            idx, tok_len, wav_path, cache_used,
                            float(wav16.min().item()) if wav16.numel() else None,
                            float(wav16.max().item()) if wav16.numel() else None,
                            float(wav16.mean().item()) if wav16.numel() else None)
                zero_count += 1
            else:
                logger.debug("idx %d: emb sum=%s tok_len=%s wav=%s cache=%s", idx, s, tok_len, wav_path, cache_used)
        total += 1

    logger.info("Checked %d items, zero embeddings: %d", total, zero_count)


if __name__ == '__main__':
    main()
