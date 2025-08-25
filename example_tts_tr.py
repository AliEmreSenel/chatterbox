"""Simple, correct TTS example that loads models on the target device and applies adapters
without ad-hoc device hacks. This script extends embeddings on the model device when needed
and applies LoRA adapters by preferring PEFT, otherwise merging LoRA into base weights.
"""
import argparse
from pathlib import Path
import torch
import torchaudio as ta

try:
    from peft import set_peft_model_state_dict, get_peft_model, LoraConfig
except Exception:
    set_peft_model_state_dict = None
    get_peft_model = None
    LoraConfig = None

from chatterbox.tts import ChatterboxTTS


def remap_keys(sd: dict, strip_tfmr_prefix: bool = False) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("tfmr.base_model.model."):
            nk = nk.replace("tfmr.base_model.model.", "tfmr.")
        elif nk.startswith("tfmr.base_model."):
            nk = nk.replace("tfmr.base_model.", "tfmr.")
        elif nk.startswith("base_model.model."):
            nk = nk.replace("base_model.model.", "")
        elif nk.startswith("base_model."):
            nk = nk.replace("base_model.", "")
        nk = nk.replace(".lora_A.default.", ".lora_A.")
        nk = nk.replace(".lora_B.default.", ".lora_B.")
        nk = nk.replace(".base_layer.", ".")
        if strip_tfmr_prefix and nk.startswith("tfmr."):
            nk = nk[len("tfmr.") :]
        out[nk] = v
    return out


def merge_lora_into_base(sd: dict, device: torch.device) -> dict:
    sd = dict(sd)
    groups = {}
    for k in list(sd.keys()):
        kk = k.replace(".default.", ".")
        if kk.endswith(".lora_A.weight"):
            prefix = kk[: -len(".lora_A.weight")]
            groups.setdefault(prefix, {})["A"] = sd[k]
        elif kk.endswith(".lora_B.weight"):
            prefix = kk[: -len(".lora_B.weight")]
            groups.setdefault(prefix, {})["B"] = sd[k]
        elif kk.endswith(".base_layer.weight"):
            prefix = kk[: -len(".base_layer.weight")]
            groups.setdefault(prefix, {})["base_layer"] = sd[k]
    for prefix, parts in groups.items():
        if "A" in parts and "B" in parts:
            A = torch.as_tensor(parts["A"]).to(device)
            B = torch.as_tensor(parts["B"]).to(device)
            try:
                delta = B.matmul(A)
            except Exception:
                delta = B @ A
            target_key = prefix + ".weight"
            base = parts.get("base_layer")
            if base is not None:
                base = torch.as_tensor(base).to(device)
                new_w = base + delta
            else:
                if target_key in sd:
                    new_w = torch.as_tensor(sd[target_key]).to(device) + delta
                else:
                    new_w = delta
            sd[target_key] = new_w
            for suf in (
                ".lora_A.weight",
                ".lora_B.weight",
                ".lora_A.default.weight",
                ".lora_B.default.weight",
                ".base_layer.weight",
            ):
                rk = prefix + suf
                sd.pop(rk, None)
            alt_prefixes = [
                "tfmr.base_model.model." + prefix,
                "tfmr.base_model." + prefix,
                "base_model.model." + prefix,
                "base_model." + prefix,
            ]
            for ap in alt_prefixes:
                for suf in (".lora_A.weight", ".lora_B.weight", ".base_layer.weight", ".lora_A.default.weight", ".lora_B.default.weight"):
                    sd.pop(ap + suf, None)
    return sd


def extend_module_rows_if_needed(module, target_rows: int):
    if module is None or not hasattr(module, "weight"):
        return
    w = module.weight
    cur_rows = w.size(0)
    if cur_rows >= target_rows:
        return
    device = w.device
    cols = w.size(1)
    new_w = torch.empty((target_rows, cols), dtype=w.dtype, device=device)
    new_w[:cur_rows].copy_(w.data)
    torch.nn.init.normal_(new_w[cur_rows:], mean=0.0, std=w.std().item() if w.numel() else 0.02)
    module.weight = torch.nn.Parameter(new_w)
    if isinstance(module, torch.nn.Embedding):
        try:
            module.num_embeddings = target_rows
        except Exception:
            pass
    if hasattr(module, "out_features"):
        try:
            module.out_features = target_rows
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="ckpt")
    p.add_argument("--adapter_dir", default="lora_tr_out")
    p.add_argument("--text", default="Merhaba! Bugün nasılsın? Bu cümle Türkçe ses sentezi için bir testtir.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device_str)

    # Load base models onto CPU first to avoid early GPU allocations/auto-offload.
    tts = ChatterboxTTS.from_local(args.ckpt_dir, "cpu")

    # If there's a base_updates.pt in the checkpoint dir, load extended text embeddings/head from it
    base_updates_path = Path(args.ckpt_dir) / "base_updates.pt"
    if base_updates_path.exists():
        bu = torch.load(str(base_updates_path), map_location="cpu")
        if isinstance(bu, dict):
            # support two layouts: {"t3_state": {...}} or flat {"text_emb": {...}, "text_head": {...}}
            if "t3_state" in bu:
                sd = bu["t3_state"]
            else:
                sd = {}
                if "text_emb" in bu:
                    te = bu["text_emb"]
                    if isinstance(te, dict) and "weight" in te:
                        sd["text_emb.weight"] = te["weight"]
                if "text_head" in bu:
                    th = bu["text_head"]
                    if isinstance(th, dict) and "weight" in th:
                        sd["text_head.weight"] = th["weight"]
                    if isinstance(th, dict) and "bias" in th:
                        sd["text_head.bias"] = th["bias"]
            if "text_emb.weight" in sd:
                w = torch.as_tensor(sd["text_emb.weight"]).clone()
                emb = torch.nn.Embedding(w.size(0), w.size(1))
                emb.weight = torch.nn.Parameter(w)
                try:
                    emb = emb.to(tts.t3.text_emb.weight.device)
                except Exception:
                    pass
                setattr(tts.t3, "text_emb", emb)
                sd.pop("text_emb.weight", None)
            if "text_head.weight" in sd:
                w = torch.as_tensor(sd["text_head.weight"]).clone()
                b = sd.get("text_head.bias", None)
                hidden = w.size(1)
                new_head = torch.nn.Linear(hidden, w.size(0), bias=(b is not None))
                new_head.weight = torch.nn.Parameter(w)
                if b is not None:
                    new_head.bias = torch.nn.Parameter(torch.as_tensor(b).clone())
                    sd.pop("text_head.bias", None)
                try:
                    new_head = new_head.to(tts.t3.text_head.weight.device)
                except Exception:
                    pass
                setattr(tts.t3, "text_head", new_head)
                sd.pop("text_head.weight", None)

    adapter_path = Path(args.adapter_dir)
    # Adapter dir may also provide base_updates.pt containing extended text_emb/text_head
    if adapter_path.exists():
        adapter_base_updates = adapter_path / "base_updates.pt"
        if adapter_base_updates.exists():
            bu = torch.load(str(adapter_base_updates), map_location="cpu")
            if isinstance(bu, dict):
                if "t3_state" in bu:
                    sd = bu["t3_state"]
                else:
                    sd = {}
                    if "text_emb" in bu:
                        te = bu["text_emb"]
                        if isinstance(te, dict) and "weight" in te:
                            sd["text_emb.weight"] = te["weight"]
                    if "text_head" in bu:
                        th = bu["text_head"]
                        if isinstance(th, dict) and "weight" in th:
                            sd["text_head.weight"] = th["weight"]
                        if isinstance(th, dict) and "bias" in th:
                            sd["text_head.bias"] = th["bias"]
                if "text_emb.weight" in sd:
                    w = torch.as_tensor(sd["text_emb.weight"]).clone()
                    emb = torch.nn.Embedding(w.size(0), w.size(1))
                    emb.weight = torch.nn.Parameter(w)
                    try:
                        emb = emb.to(tts.t3.text_emb.weight.device)
                    except Exception:
                        pass
                    setattr(tts.t3, "text_emb", emb)
                    sd.pop("text_emb.weight", None)
                if "text_head.weight" in sd:
                    w = torch.as_tensor(sd["text_head.weight"]).clone()
                    b = sd.get("text_head.bias", None)
                    hidden = w.size(1)
                    new_head = torch.nn.Linear(hidden, w.size(0), bias=(b is not None))
                    new_head.weight = torch.nn.Parameter(w)
                    if b is not None:
                        new_head.bias = torch.nn.Parameter(torch.as_tensor(b).clone())
                        sd.pop("text_head.bias", None)
                    try:
                        new_head = new_head.to(tts.t3.text_head.weight.device)
                    except Exception:
                        pass
                    setattr(tts.t3, "text_head", new_head)
                    sd.pop("text_head.weight", None)

        ckpt_file = adapter_path / "checkpoint.pt"
        if ckpt_file.exists():
            ck = torch.load(str(ckpt_file), map_location="cpu")
            if isinstance(ck, dict):
                # extend embeddings/heads on their model device before loading
                if "t3_state" in ck:
                    sd = ck["t3_state"]
                    # If checkpoint contains fully extended embeddings/heads, load them directly onto the model
                    if "text_emb.weight" in sd:
                        w = torch.as_tensor(sd["text_emb.weight"]).clone()
                        emb = torch.nn.Embedding(w.size(0), w.size(1))
                        emb.weight = torch.nn.Parameter(w)
                        try:
                            emb = emb.to(tts.t3.text_emb.weight.device)
                        except Exception:
                            pass
                        setattr(tts.t3, "text_emb", emb)
                        sd.pop("text_emb.weight", None)
                    if "text_head.weight" in sd:
                        w = torch.as_tensor(sd["text_head.weight"]).clone()
                        b = sd.get("text_head.bias", None)
                        hidden = w.size(1)
                        new_head = torch.nn.Linear(hidden, w.size(0), bias=(b is not None))
                        new_head.weight = torch.nn.Parameter(w)
                        if b is not None:
                            new_head.bias = torch.nn.Parameter(torch.as_tensor(b).clone())
                            sd.pop("text_head.bias", None)
                        try:
                            new_head = new_head.to(tts.t3.text_head.weight.device)
                        except Exception:
                            pass
                        setattr(tts.t3, "text_head", new_head)
                        sd.pop("text_head.weight", None)
                    tts.t3.load_state_dict(sd, strict=False)
                if "tfmr_state" in ck and hasattr(tts.t3, "tfmr"):
                    sd = ck["tfmr_state"]
                    # prefer PEFT helper when available and applicable
                    applied = False
                    if set_peft_model_state_dict is not None and getattr(tts.t3.tfmr, "peft_config", None) is not None:
                        try:
                            set_peft_model_state_dict(tts.t3.tfmr, sd)
                            applied = True
                        except Exception:
                            applied = False
                    if not applied:
                        merged = merge_lora_into_base(sd, device)
                        remapped = remap_keys(merged, strip_tfmr_prefix=True)
                        tts.t3.tfmr.load_state_dict(remapped, strict=False)
                if "s3gen_state" in ck and hasattr(tts, "s3gen"):
                    tts.s3gen.load_state_dict(ck["s3gen_state"], strict=False)
        else:
            # individual files
            t3_lora = adapter_path / "t3_lora_finetuned.pt"
            t3_tfmr = adapter_path / "t3_tfmr_lora_finetuned.pt"
            s3_lora = adapter_path / "s3gen_lora_finetuned.pt"
            if t3_lora.exists():
                sd = torch.load(str(t3_lora), map_location=device)
                # extend if needed
                for key in ("text_emb.weight", "text_head.weight"):
                    if key in sd:
                        rows = int(sd[key].shape[0])
                        mod_name, _ = key.rsplit(".", 1)
                        mod = getattr(tts.t3, mod_name, None)
                        extend_module_rows_if_needed(mod, rows)
                tts.t3.load_state_dict(sd, strict=False)
            if t3_tfmr.exists() and hasattr(tts.t3, "tfmr"):
                sd = torch.load(str(t3_tfmr), map_location=device)
                applied = False
                if set_peft_model_state_dict is not None and getattr(tts.t3.tfmr, "peft_config", None) is not None:
                    try:
                        set_peft_model_state_dict(tts.t3.tfmr, sd)
                        applied = True
                    except Exception:
                        applied = False
                if not applied:
                    merged = merge_lora_into_base(sd, device)
                    remapped = remap_keys(merged, strip_tfmr_prefix=True)
                    tts.t3.tfmr.load_state_dict(remapped, strict=False)
            if s3_lora.exists() and hasattr(tts, "s3gen"):
                sd = torch.load(str(s3_lora), map_location=device)
                tts.s3gen.load_state_dict(sd, strict=False)

    print("Generating...")
    wav = tts.generate(args.text)

    if isinstance(wav, torch.Tensor):
        wav_out = wav.squeeze(0).cpu()
    else:
        import numpy as _np

        wav_out = torch.from_numpy(_np.asarray(wav)).squeeze(0)

    if wav_out.ndim == 1:
        wav_out = wav_out.unsqueeze(0)
    wav_out = wav_out.to(dtype=torch.float32)
    ta.save("tts_tr_example.wav", wav_out, sample_rate=tts.sr)
    print("Saved tts_tr_example.wav")
