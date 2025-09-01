"""Simple, correct TTS example that loads models on the target device and applies adapters
without ad-hoc device hacks. This script extends embeddings on the model device when needed
and applies LoRA adapters by preferring PEFT, otherwise merging LoRA into base weights.
"""

import argparse
from pathlib import Path
import logging
import sys

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def trace_calls(frame, event, arg):
    if event == "call":
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        print(f"Call to {func_name} in {filename}:{lineno}")
    return trace_calls


# Enable tracing
# sys.settrace(trace_calls)

logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _torch_load_verbose(path: Path, map_location="cpu"):
    try:
        sd = torch.load(str(path), map_location=map_location)
        try:
            # basic stats
            if isinstance(sd, dict):
                kcnt = len(sd.keys())
                lora_cnt = sum(
                    1
                    for k in sd.keys()
                    if "lora_A" in k or "lora_B" in k or ".lora_" in k
                )
                tfmr_cnt = sum(
                    1 for k in sd.keys() if k.startswith("tfmr.") or "tfmr" in k
                )
                logger.info(
                    "Loaded checkpoint '%s' (keys=%d, lora_keys=%d, tfmr_keys=%d)",
                    path,
                    kcnt,
                    lora_cnt,
                    tfmr_cnt,
                )
        except Exception:
            pass
        return sd
    except Exception as e:
        msg = f"Failed to load checkpoint '{path}': {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="ckpt")
    p.add_argument("--adapter_dir", default="lora_tr_out")
    p.add_argument(
        "--text",
        default="Merhaba! Bugün nasılsın? Bu cümle Türkçe ses sentezi için bir testtir.",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device_str = args.device or (
        "cpu"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device = torch.device(device_str)

    tts = ChatterboxTTS.from_local(args.ckpt_dir, "cpu")

    base_updates_path = Path(args.adapter_dir) / "base_updates.pt"

    bu = _torch_load_verbose(base_updates_path, map_location="cpu")

    w = bu["text_emb"]["weight"]
    emb = torch.nn.Embedding(w.size(0), w.size(1))
    emb.weight = torch.nn.Parameter(w)
    emb = emb.to(tts.t3.text_emb.weight.device)
    setattr(tts.t3, "text_emb", emb)

    w = bu["text_head"]["weight"]
    new_head = torch.nn.Linear(w.size(1), w.size(0))
    new_head.weight = torch.nn.Parameter(w)
    new_head = new_head.to(tts.t3.text_head.weight.device)
    setattr(tts.t3, "text_head", new_head)

    # wrap t3
    lora_cfg_t3 = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "donw_proj",
        ],
        inference_mode=True,
    )

    tts.t3 = get_peft_model(tts.t3, lora_cfg_t3)
    logger.info("Wrapped tts.t3 with PEFT for LoRA application")

    adapter_path = Path(args.adapter_dir)
    if adapter_path.exists():
        # apply t3 LoRA
        t3_lora = adapter_path / "lora_t3_state.pt"
        if t3_lora.exists():
            sd = _torch_load_verbose(t3_lora, map_location="cpu")
            print(sd)
            logger.info("Applying t3 LoRA from %s via PEFT", t3_lora)
            set_peft_model_state_dict(tts.t3, sd)
        # apply s3gen LoRA
        s3_lora = adapter_path / "s3gen_lora_finetuned.pt"
        if s3_lora.exists():
            lora_cfg_s3 = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules="all-linear",
                inference_mode=True,
            )

            tts.s3gen = get_peft_model(tts.s3gen, lora_cfg_s3)

            logger.info("Wrapped tts.s3gen with PEFT for LoRA application")
            sd = _torch_load_verbose(s3_lora, map_location="cpu")
            logger.info("Applying s3gen LoRA from %s via PEFT", s3_lora)
            set_peft_model_state_dict(tts.s3gen, sd)

    # Move models to target device before generation
    tts.device = device
    tts.t3.to(device)
    tts.s3gen.to(device)
    tts.ve.to(device)
    tts.conds = tts.conds.to(device)

    logger.info("Generating...")
    try:
        wav = tts.generate(args.text)
    except Exception as e:
        msg = f"Generation failed: {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e

    ta.save("tts_tr_example.wav", wav, sample_rate=tts.sr)

    logger.info("Saved tts_tr_example.wav")
