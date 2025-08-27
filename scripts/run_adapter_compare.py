#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import copy
from chatterbox.tts import ChatterboxTTS, punc_norm
import importlib.util, sys
spec = importlib.util.spec_from_file_location("example_tts_tr", str(Path(__file__).parent.parent / "example_tts_tr.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["example_tts_tr"] = mod
spec.loader.exec_module(mod)
_torch_load_verbose = mod._torch_load_verbose
_safe_apply_to_target = mod._safe_apply_to_target


def run_one(tts: ChatterboxTTS, text: str, max_new_tokens=300):
    if tts.conds is None:
        try:
            tts.prepare_conditionals("src/chatterbox/test-1.wav")
        except Exception:
            try:
                tts.prepare_conditionals(Path(__file__).parent.parent / "src/chatterbox/test-1.wav")
            except Exception:
                pass
    try:
        emb_dev = next(tts.t3.text_emb.parameters()).device
    except Exception:
        emb_dev = torch.device(tts.device)
    text_tokens = tts.tokenizer.text_to_tokens(punc_norm(text)).to(emb_dev)
    sot = tts.t3.hp.start_text_token
    eot = tts.t3.hp.stop_text_token
    text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
    text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)
    with torch.inference_mode():
        speech_tokens = tts.t3.inference(t3_cond=tts.conds.t3, text_tokens=text_tokens, max_new_tokens=max_new_tokens)
    speech_tokens = speech_tokens[0].detach().cpu().numpy()
    wav = None
    try:
        st = torch.from_numpy(speech_tokens).to(tts.device)
        wav, _ = tts.s3gen.inference(speech_tokens=st, ref_dict=tts.conds.gen)
        wav = wav.squeeze(0).detach().cpu().numpy()
    except Exception:
        wav = None
    return speech_tokens, wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="ckpt")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--out_dir", default="adapter_dumps")
    p.add_argument("--text", default="Merhaba! Bugün nasılsın? Bu cümle Türkçe ses sentezi için bir testtir.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model for baseline...")
    tts = ChatterboxTTS.from_local(args.ckpt_dir, args.device)
    tts.t3.to(args.device)
    if hasattr(tts, "s3gen") and tts.s3gen is not None:
        tts.s3gen.to(args.device)
    print("Running baseline generation (no adapters)...")
    toks, wav = run_one(tts, args.text)
    np.save(out_dir / "baseline_tokens.npy", toks)
    if wav is not None:
        torchaudio.save(str(out_dir / "baseline.wav"), torch.from_numpy(wav).unsqueeze(0), sample_rate=tts.sr)
    print("Saved baseline tokens and wav")

    # Save baseline state dicts for restoring before each adapter apply
    base_t3_sd = {k: v.cpu().clone() for k, v in tts.t3.state_dict().items()}
    base_s3_sd = None
    if hasattr(tts, "s3gen") and tts.s3gen is not None:
        base_s3_sd = {k: v.cpu().clone() for k, v in tts.s3gen.state_dict().items()}

    if args.adapter_dir:
        adapter_path = Path(args.adapter_dir)
        if not adapter_path.exists():
            print("Adapter dir not found, skipping adapters")
            return
        files = list(adapter_path.glob("*.pt")) + list(adapter_path.glob("*.safetensors"))
        files = sorted(files, key=lambda p: p.stat().st_mtime)
        for f in files:
            name = f.stem
            print("Processing adapter", f)
            try:
                sd = _torch_load_verbose(f, map_location="cpu")
            except Exception as e:
                print("Failed to load adapter", f, e)
                continue
            # Restore base weights into live model before applying adapter
            try:
                tts.t3.load_state_dict(base_t3_sd, strict=False)
            except Exception as e:
                print("Warning: failed to restore base t3 state before applying adapter:", e)
            if base_s3_sd is not None and hasattr(tts, "s3gen") and tts.s3gen is not None:
                try:
                    tts.s3gen.load_state_dict(base_s3_sd, strict=False)
                except Exception as e:
                    print("Warning: failed to restore base s3 state before applying adapter:", e)
            # move to device
            try:
                tts.t3.to(args.device)
            except Exception:
                pass
            if hasattr(tts, "s3gen") and tts.s3gen is not None:
                try:
                    tts.s3gen.to(args.device)
                except Exception:
                    pass
            applied = False
            last_exc = None
            for target in ("s3gen", "t3", "tfmr"):
                try:
                    _safe_apply_to_target(sd, target, tts, torch.device(args.device), label=name)
                    applied = True
                    target_used = target
                    break
                except Exception as e:
                    last_exc = e
            if not applied:
                print("Failed to apply adapter", f, "error:", last_exc)
                continue
            print("Applied adapter", f, "to target", target_used)
            try:
                toks, wav = run_one(tts, args.text)
            except Exception as e:
                print("Generation failed for adapter", f, e)
                toks = None
                wav = None
            if toks is not None:
                np.save(out_dir / f"{name}_tokens.npy", toks)
            if wav is not None:
                try:
                    torchaudio.save(str(out_dir / f"{name}.wav"), torch.from_numpy(wav).unsqueeze(0), sample_rate=tts.sr)
                except Exception as e:
                    print("Failed saving wav for", f, e)
            print("Done", f)

if __name__ == "__main__":
    main()
