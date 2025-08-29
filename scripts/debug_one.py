#!/usr/bin/env python3
import argparse
import traceback
import logging
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.models.s3gen import S3Gen

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

p = argparse.ArgumentParser()
p.add_argument('--wav', required=True)
p.add_argument('--ckpt_dir', required=False)
args = p.parse_args()

wav_path = args.wav
s3gen = S3Gen()
if args.ckpt_dir:
    try:
        from safetensors.torch import load_file

        state = load_file(Path(args.ckpt_dir) / 's3gen.safetensors')
        s3gen.load_state_dict(state, strict=False)
        logger.info('Loaded checkpoint')
    except Exception:
        logger.exception('Failed to load checkpoint')

try:
    waveform, sr = ta.load(wav_path)
except Exception:
    logger.exception('Failed to load wav')
    raise
if waveform.dim() > 1:
    waveform = waveform.mean(dim=0)
if sr != 16000:
    waveform = ta.transforms.Resample(sr, 16000)(waveform.unsqueeze(0)).squeeze(0)
wav16 = waveform

try:
    emb = s3gen.speaker_encoder.inference(wav16.unsqueeze(0))
    logger.info('emb type=%s dim=%s', type(emb), None if not hasattr(emb,'dim') else emb.dim())
    logger.info('emb repr=%s', repr(emb)[:200])
except Exception:
    logger.error('speaker_encoder.inference raised:')
    traceback.print_exc()
    emb = None

print('DONE, emb is', emb)
