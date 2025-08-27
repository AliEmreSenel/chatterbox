"""Simple, correct TTS example that loads models on the target device and applies adapters
without ad-hoc device hacks. This script extends embeddings on the model device when needed
and applies LoRA adapters by preferring PEFT, otherwise merging LoRA into base weights.
"""

import argparse
from pathlib import Path
import logging
import sys

try:
    import torch
except Exception as e:
    torch = None
    logging.exception("Required dependency 'torch' is not available: %s", e)
try:
    import torchaudio as ta
except Exception as e:
    ta = None
    logging.exception("Required dependency 'torchaudio' is not available: %s", e)

if torch is None:
    msg = "PyTorch (torch) is required to run this script. Install it from https://pytorch.org/"
    print(msg, file=sys.stderr)
    raise RuntimeError(msg)
if ta is None:
    msg = "torchaudio is required to save generated audio. Install it with 'pip install torchaudio'"
    print(msg, file=sys.stderr)
    raise RuntimeError(msg)

try:
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
except Exception as e:
    logging.warning("PEFT import failed: %s", e)
    set_peft_model_state_dict = None
    get_peft_model = None
    LoraConfig = None

from chatterbox.tts import ChatterboxTTS, punc_norm
import copy


def _dry_run_and_apply(
    sd: dict,
    target: str,
    tts: "ChatterboxTTS",
    device: torch.device,
    label: str = "adapter",
    test_text: str = "Merhaba.",
) -> None:
    """Dry-run applying sd to a deepcopy of tts and, if successful, apply to real tts.

    target: one of 't3', 'tfmr', 's3gen'
    """
    logger.info("Starting dry-run for %s -> target=%s", label, target)
    try:
        tts_copy = copy.deepcopy(tts)
    except Exception as e:
        logger.warning("Deepcopy of tts failed, skipping dry-run: %s", e)
        # fall back to direct apply
        _safe_apply_to_target(sd, target, tts, device, label)
        return
    try:
        # apply to copy
        _safe_apply_to_target(sd, target, tts_copy, device, label)
    except Exception as e:
        logger.exception("Dry-run apply failed for %s: %s", label, e)
        raise RuntimeError(f"Dry-run validation failed for {label}: {e}") from e
    # run a short generation
    try:
        logger.info("Running short generation on dry-run copy for %s", label)
        out = tts_copy.generate(test_text, max_new_tokens=64)
        logger.info("Dry-run generation succeeded for %s", label)
    except Exception as e:
        logger.exception("Dry-run generation failed for %s: %s", label, e)
        raise RuntimeError(f"Dry-run generation failed for {label}: {e}") from e
    # if reached here, apply to real tts
    logger.info("Dry-run validation passed for %s, applying to live model", label)
    _safe_apply_to_target(sd, target, tts, device, label)


def _safe_apply_to_target(
    sd: dict,
    target: str,
    tts: "ChatterboxTTS",
    device: torch.device,
    label: str = "adapter",
) -> None:
    """Apply sd to tts with validation. target in ('t3','tfmr','s3gen').

    Heuristic: if the adapter state dict clearly targets T3 (text_emb/text_head/tfmr
    keys) but the caller requested 's3gen', prefer applying to T3 to avoid accidentally
    loading T3 weights into the s3gen module.
    """
    # basic key list
    keys = list(sd.keys()) if isinstance(sd, dict) else []

    def _looks_like_t3(keys_list):
        for k in keys_list:
            if (
                "text_emb" in k
                or "text_head" in k
                or "tfmr" in k
                or k.startswith("t3")
                or "speech_pos_emb" in k
            ):
                return True
        return False

    def _looks_like_s3(keys_list):
        for k in keys_list:
            if (
                "flow" in k
                or "decoder" in k
                or "input_embedding" in k
                or "s3gen" in k
                or "hifigan" in k
                or "speech_feat" in k
            ):
                return True
        return False

    # If caller requested s3gen but state dict looks like T3, switch to t3
    if target == "s3gen" and _looks_like_t3(keys) and not _looks_like_s3(keys):
        logger.info(
            "Adapter appears to target T3 (keys indicate t3); applying to 't3' instead of 's3gen'"
        )
        target = "t3"

    if target == "t3":
        # if lora-like keys, merge then apply
        is_lora_state = any(
            "lora_A" in k or "lora_B" in k or ".lora_" in k for k in keys
        )
        if is_lora_state:
            merged = merge_lora_into_base(sd, device)
            try:
                _safe_apply_state_dict(merged, tts.t3, device, prefix_desc=label)
            except Exception:
                # try applying into transformer if present
                remapped = remap_keys(merged, strip_tfmr_prefix=True)
                _safe_apply_state_dict(
                    remapped, getattr(tts.t3, "tfmr"), device, prefix_desc=label
                )
        else:
            # try PEFT then plain
            if (
                set_peft_model_state_dict is not None
                and getattr(tts.t3, "peft_config", None) is not None
            ):
                try:
                    set_peft_model_state_dict(tts.t3, sd)
                    return
                except Exception:
                    pass
            _safe_apply_state_dict(sd, tts.t3, device, prefix_desc=label)
    elif target == "tfmr":
        # target transformer specifically
        if (
            set_peft_model_state_dict is not None
            and getattr(tts.t3, "tfmr", None) is not None
            and getattr(tts.t3.tfmr, "peft_config", None) is not None
        ):
            try:
                set_peft_model_state_dict(tts.t3.tfmr, sd)
                return
            except Exception:
                pass
        # merge if lora-like
        if any("lora_A" in k or "lora_B" in k or ".lora_" in k for k in keys):
            merged = merge_lora_into_base(sd, device)
            remapped = remap_keys(merged, strip_tfmr_prefix=True)
            _safe_apply_state_dict(
                remapped, getattr(tts.t3, "tfmr"), device, prefix_desc=label
            )
        else:
            _safe_apply_state_dict(
                sd, getattr(tts.t3, "tfmr"), device, prefix_desc=label
            )
    elif target == "s3gen":
        # s3gen application
        if any("lora_A" in k or "lora_B" in k or ".lora_" in k for k in keys):
            # try PEFT
            if (
                set_peft_model_state_dict is not None
                and getattr(tts.s3gen, "peft_config", None) is not None
            ):
                try:
                    set_peft_model_state_dict(tts.s3gen, sd)
                    return
                except Exception:
                    pass
            merged = merge_lora_into_base(sd, device)
            remapped = remap_keys(merged, strip_tfmr_prefix=False)
            _safe_apply_state_dict(remapped, tts.s3gen, device, prefix_desc=label)
        else:
            _safe_apply_state_dict(sd, tts.s3gen, device, prefix_desc=label)
    else:
        raise RuntimeError(f"Unknown target for apply: {target}")


logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _apply_peft_model(model, sd, target_modules):
    """Wrap `model` with PEFT using `target_modules` and apply PEFT state dict `sd`.
    Returns the wrapped model on success, raises on failure.
    """
    if (
        get_peft_model is None
        or set_peft_model_state_dict is None
        or LoraConfig is None
    ):
        raise RuntimeError("PEFT utilities are not available")
    cfg = LoraConfig(
        r=8, lora_alpha=32, target_modules=target_modules, inference_mode=False
    )
    wrapped = get_peft_model(model, cfg)
    set_peft_model_state_dict(wrapped, sd)
    return wrapped


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
            A_raw = parts["A"]
            B_raw = parts["B"]
            A = torch.as_tensor(A_raw).to(device)
            B = torch.as_tensor(B_raw).to(device)
            delta = None
            # Try common linear case first
            try:
                if A.ndim == 2 and B.ndim == 2:
                    delta = B.matmul(A)
                else:
                    # Conv / higher-dim cases: attempt to handle shapes like
                    # A: (r, in_channels, kernel...), B: (out_channels, r, maybe 1)
                    # Try squeezing singleton trailing dims on B
                    B_s = B
                    if B_s.ndim > 2 and B_s.shape[-1] == 1:
                        B_s = B_s.squeeze(-1)
                    # If B_s is (out, r) and A is (r, in, k...)
                    if B_s.ndim == 2 and A.ndim >= 2 and A.shape[0] == B_s.shape[1]:
                        # einsum over rank dim
                        # build subscripts dynamically: A dims = (r, d1, d2, ...)
                        # we want delta dims = (out, d1, d2, ...)
                        r = A.shape[0]
                        out = B_s.shape[0]
                        # flatten remaining A dims into one for matmul then reshape
                        rem = int(A.numel() // r)
                        A_flat = A.reshape(r, rem)
                        try:
                            delta_flat = B_s.matmul(A_flat)
                            delta = delta_flat.reshape(out, *A.shape[1:])
                        except Exception:
                            # fallback to einsum
                            try:
                                # 'or, rik -> oik' pattern generalized
                                # construct einsum string
                                a_subs = "".join(
                                    chr(ord("i") + i) for i in range(A.ndim - 1)
                                )
                                eins_in = f"or,{ 'r' + a_subs } -> o{a_subs}"
                                delta = torch.einsum(eins_in, B_s, A)
                            except Exception as ee:
                                logging.warning(
                                    "Einsum fallback failed for prefix %s: %s",
                                    prefix,
                                    ee,
                                )
                    # If still None, try simple @ with broadcasting
                    if delta is None:
                        try:
                            delta = B @ A
                        except Exception as e:
                            logging.warning(
                                "Failed to compute delta for %s with B@A: %s", prefix, e
                            )
                            # try transposed variants
                            try:
                                delta = B.t().matmul(A.t()).t()
                            except Exception:
                                pass
                if delta is None:
                    raise RuntimeError(
                        f"Could not compute LoRA delta for prefix {prefix} (shapes A={tuple(A.shape)}, B={tuple(B.shape)})"
                    )
            except Exception as e:
                # bubble up with context
                raise RuntimeError(
                    f"Failed while merging LoRA for {prefix}: {e}"
                ) from e
            target_key = prefix + ".weight"
            base = parts.get("base_layer")
            if base is not None:
                base = torch.as_tensor(base).to(device)
                if base.shape != delta.shape:
                    # try to broadcast/reshape if possible
                    try:
                        if base.numel() == delta.numel():
                            new_w = base.view_as(delta) + delta
                        else:
                            raise RuntimeError(
                                f"Base weight shape {tuple(base.shape)} incompatible with delta {tuple(delta.shape)}"
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f"Shape mismatch when adding LoRA delta to base for {prefix}: {e}"
                        ) from e
                else:
                    new_w = base + delta
            else:
                if target_key in sd:
                    base_candidate = torch.as_tensor(sd[target_key]).to(device)
                    if base_candidate.shape == delta.shape:
                        new_w = base_candidate + delta
                    else:
                        # if shapes differ but total elements match, reshape
                        if base_candidate.numel() == delta.numel():
                            new_w = base_candidate.view_as(delta) + delta
                        else:
                            raise RuntimeError(
                                f"No compatible base found for {prefix}; base shape {tuple(base_candidate.shape)} vs delta {tuple(delta.shape)}"
                            )
                else:
                    new_w = delta
            sd[target_key] = new_w
            # remove LoRA keys
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
                for suf in (
                    ".lora_A.weight",
                    ".lora_B.weight",
                    ".base_layer.weight",
                    ".lora_A.default.weight",
                    ".lora_B.default.weight",
                ):
                    sd.pop(ap + suf, None)
    return sd


def _get_state_shapes(sd: dict) -> dict:
    out = {}
    try:
        import numpy as _np
    except Exception:
        _np = None
    if not isinstance(sd, dict):
        return out
    for k, v in sd.items():
        try:
            if hasattr(v, "shape"):
                out[k] = tuple(v.shape)
            else:
                if _np is not None:
                    out[k] = tuple(_np.asarray(v).shape)
                else:
                    out[k] = None
        except Exception:
            out[k] = None
    return out


def _report_state_dict_changes(before: dict, after_sd: dict, context: str):
    try:
        after = _get_state_shapes(after_sd)
    except Exception:
        after = {}
    before = before or {}
    added = [k for k in after.keys() if k not in before]
    removed = [k for k in before.keys() if k not in after]
    shape_changes = []
    for k in before.keys():
        if k in after:
            if before.get(k) != after.get(k):
                shape_changes.append((k, before.get(k), after.get(k)))
    if added:
        logger.info("%s: %d keys added: %s", context, len(added), added[:10])
    if removed:
        logger.warning("%s: %d keys removed: %s", context, len(removed), removed[:10])
    for k, b, a in shape_changes:
        logger.warning("%s: key '%s' shape changed from %s to %s", context, k, b, a)
    if not (added or removed or shape_changes):
        logger.debug("%s: no state-dict key/shape changes detected", context)
    return {"added": added, "removed": removed, "shape_changes": shape_changes}


def apply_with_report(sd: dict, module, device, name: str):
    # strict apply: directly use module.load_state_dict(sd, strict=True) and let it raise on mismatches
    logger.info("Applying state dict to module %s (strict)", name)
    module.load_state_dict(sd, strict=True)
    return {"applied": True}


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
    torch.nn.init.normal_(
        new_w[cur_rows:], mean=0.0, std=w.std().item() if w.numel() else 0.02
    )
    module.weight = torch.nn.Parameter(new_w)
    if isinstance(module, torch.nn.Embedding):
        try:
            module.num_embeddings = target_rows
        except Exception as e:
            logging.warning(
                "Failed to set num_embeddings on module %s: %s",
                getattr(module, "__class__", type(module)),
                e,
            )
    if hasattr(module, "out_features"):
        try:
            module.out_features = target_rows
        except Exception as e:
            logging.warning(
                "Failed to set out_features on module %s: %s",
                getattr(module, "__class__", type(module)),
                e,
            )


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


def _module_param_stats(module, filters=None, max_items=5):
    out = []
    try:
        for n, p in module.named_parameters():
            if filters:
                ok = False
                for f in filters:
                    if f in n:
                        ok = True
                        break
                if not ok:
                    continue
            try:
                out.append((n, float(torch.norm(p).item())))
            except Exception:
                try:
                    out.append((n, float(p.detach().cpu().abs().sum().item())))
                except Exception:
                    out.append((n, None))
            if len(out) >= max_items:
                break
    except Exception:
        return out
    return out


def _validate_tensor_array_ok(v, name="tensor"):
    try:
        t = torch.as_tensor(v)
    except Exception as e:
        raise RuntimeError(f"Adapter contains non-tensor value for {name}: {e}") from e
    if not torch.isfinite(t).all():
        raise RuntimeError(f"Adapter tensor {name} contains non-finite values")
    norm = float(torch.norm(t).item()) if t.numel() else 0.0
    if norm == 0.0:
        # zero norm might be suspicious but not fatal
        logging.warning("Adapter tensor %s has zero norm", name)
    if norm > 1e8:
        raise RuntimeError(f"Adapter tensor {name} has abnormally large norm: {norm}")
    return True


def validate_state_dict_against_module(
    sd: dict, module, allow_expand_embeddings: bool = True
):
    """Validate that state dict `sd` is compatible with `module`'s state dict.
    - ensures overlapping keys have matching shapes
    - checks numeric sanity (finite, reasonable norm)
    - allows expanding embeddings if allow_expand_embeddings is True
    Raises RuntimeError on failure.
    """
    if not isinstance(sd, dict):
        raise RuntimeError("State dict is not a mapping")
    target_sd = {
        k: v.shape if hasattr(v, "shape") else None
        for k, v in module.state_dict().items()
    }
    for k, v in sd.items():
        # accept nested wrappers where sd may include dicts like {'t3_state': {...}}
        if isinstance(v, dict) and (k.endswith("_state") or k.endswith("_dict")):
            # skip nested container at this level
            continue
        name = k
        # sanity check numeric
        try:
            _validate_tensor_array_ok(v, name=name)
        except Exception as e:
            raise RuntimeError(f"Invalid tensor for key {name}: {e}") from e
        # shape checks
        if name in target_sd and target_sd[name] is not None:
            tgt_shape = tuple(target_sd[name])
            try:
                v_shape = tuple(v.shape)
            except Exception:
                try:
                    v_shape = tuple(torch.as_tensor(v).shape)
                except Exception:
                    v_shape = None
            if v_shape is None:
                logging.warning("Could not determine shape for adapter key %s", name)
            else:
                # allow embedding row expansion
                if name.endswith("text_emb.weight") or name.endswith(
                    "text_head.weight"
                ):
                    if allow_expand_embeddings:
                        if v_shape[1] != tgt_shape[1]:
                            raise RuntimeError(
                                f"Adapter weight {name} has incompatible hidden dim {v_shape[1]} vs model {tgt_shape[1]}"
                            )
                    else:
                        if v_shape != tgt_shape:
                            raise RuntimeError(
                                f"Adapter weight {name} shape mismatch: {v_shape} vs {tgt_shape}"
                            )
                else:
                    if v_shape != tgt_shape:
                        raise RuntimeError(
                            f"Adapter key {name} shape mismatch: {v_shape} vs {tgt_shape}"
                        )
    return True


def _safe_apply_state_dict(sd: dict, target_module, device, prefix_desc="adapter"):
    # Strict apply: validate and then call load_state_dict(strict=True)
    validate_state_dict_against_module(sd, target_module, allow_expand_embeddings=False)
    target_module.load_state_dict(sd, strict=True)
    return True


# remove safe loader; use strict load directly
def safe_load_state_dict_with_report(
    sd: dict, target_module, device, prefix_desc="adapter"
):
    # No safety: load strictly and allow exceptions to propagate
    logger.info("Loading state dict into module (strict): %s", prefix_desc)
    target_module.load_state_dict(sd, strict=True)
    return True


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

    try:
        tts = ChatterboxTTS.from_local(args.ckpt_dir, "cpu")
    except Exception as e:
        msg = f"Failed to load ChatterboxTTS from '{args.ckpt_dir}': {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e

    base_updates_path = Path(args.ckpt_dir) / "base_updates.pt"
    if base_updates_path.exists():
        bu = _torch_load_verbose(base_updates_path, map_location="cpu")
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
            try:
                if "text_emb.weight" in sd:
                    w = torch.as_tensor(sd["text_emb.weight"]).clone()
                    emb = torch.nn.Embedding(w.size(0), w.size(1))
                    emb.weight = torch.nn.Parameter(w)
                    try:
                        emb = emb.to(tts.t3.text_emb.weight.device)
                    except Exception as e:
                        logger.warning("Failed to move text_emb to device: %s", e)
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
                    except Exception as e:
                        logger.warning("Failed to move text_head to device: %s", e)
                    setattr(tts.t3, "text_head", new_head)
                    logger.info(
                        "Applied text_head to tts.t3; out_features=%s in_features=%s bias=%s",
                        getattr(new_head, "out_features", None)
                        or new_head.weight.size(0),
                        getattr(new_head, "in_features", None)
                        or new_head.weight.size(1),
                        getattr(new_head, "bias", None) is not None,
                    )
                    sd.pop("text_head.weight", None)
            except Exception as e:
                msg = f"Failed while applying base_updates from '{base_updates_path}': {e}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

    # After applying base_updates from ckpt, now wrap models with PEFT and apply LoRA adapters from adapter_dir
    # Wrap with same configs used during training
    if (
        get_peft_model is None
        or LoraConfig is None
        or set_peft_model_state_dict is None
    ):
        raise RuntimeError("PEFT utilities are required to apply LoRA adapters")
    # wrap t3
    lora_cfg_t3 = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], inference_mode=False
    )
    try:
        tts.t3.to("cpu")
    except Exception:
        pass
    tts.t3 = get_peft_model(tts.t3, lora_cfg_t3)
    logger.info("Wrapped tts.t3 with PEFT for LoRA application")
    # wrap s3gen if present
    if hasattr(tts, "s3gen") and tts.s3gen is not None:
        lora_cfg_s3 = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["proj", "linear", "conv", "proj_linear"],
            inference_mode=False,
        )
        try:
            tts.s3gen.to("cpu")
        except Exception:
            pass
        tts.s3gen = get_peft_model(tts.s3gen, lora_cfg_s3)
        logger.info("Wrapped tts.s3gen with PEFT for LoRA application")

    adapter_path = Path(args.adapter_dir)
    if adapter_path.exists():
        # apply t3 LoRA
        t3_lora = adapter_path / "t3_lora_finetuned.pt"
        if t3_lora.exists():
            sd = _torch_load_verbose(t3_lora, map_location="cpu")
            if isinstance(sd, dict):
                if "t3_state" in sd:
                    sd = sd["t3_state"]
                logger.info("Applying t3 LoRA from %s via PEFT", t3_lora)
                set_peft_model_state_dict(tts.t3, sd)
        # apply transformer-specific LoRA if present
        t3_tfmr = adapter_path / "t3_tfmr_lora_finetuned.pt"
        if t3_tfmr.exists():
            sd = _torch_load_verbose(t3_tfmr, map_location="cpu")
            if isinstance(sd, dict):
                logger.info("Applying t3 transformer LoRA from %s via PEFT", t3_tfmr)
                set_peft_model_state_dict(tts.t3, sd)
        # apply s3gen LoRA
        s3_lora = adapter_path / "s3gen_lora_finetunedd.pt"
        if s3_lora.exists() and hasattr(tts, "s3gen"):
            sd = _torch_load_verbose(s3_lora, map_location="cpu")
            if isinstance(sd, dict):
                logger.info("Applying s3gen LoRA from %s via PEFT", s3_lora)
                set_peft_model_state_dict(tts.s3gen, sd)

    # Sanity checks before generation: ensure tokenizer and model heads/embeddings align
    try:
        tok_vocab = None
        try:
            tok_vocab = len(tts.tokenizer.tokenizer.get_vocab())
        except Exception:
            try:
                tok_vocab = getattr(
                    tts.tokenizer.tokenizer, "get_vocab_size", lambda: None
                )()
            except Exception:
                tok_vocab = None
        if tok_vocab is not None:
            # check text_emb
            emb_layer = getattr(tts.t3, "text_emb", None)
            if emb_layer is not None and hasattr(emb_layer, "weight"):
                cur_size = getattr(emb_layer, "num_embeddings", None)
                if cur_size is None and hasattr(emb_layer, "weight"):
                    cur_size = emb_layer.weight.size(0)
                if cur_size is not None and cur_size < tok_vocab:
                    logger.warning(
                        "Extending text_emb from %d to tokenizer vocab %d",
                        cur_size,
                        tok_vocab,
                    )
                    extend_module_rows_if_needed(emb_layer, int(tok_vocab))
            # check text_head
            head = getattr(tts.t3, "text_head", None)
            if head is not None and hasattr(head, "weight"):
                out_features = getattr(head, "out_features", None)
                if out_features is None:
                    try:
                        out_features = head.weight.size(0)
                    except Exception:
                        out_features = None
                if out_features is not None and out_features < tok_vocab:
                    logger.warning(
                        "Extending text_head from %d to tokenizer vocab %d",
                        out_features,
                        tok_vocab,
                    )
                    # create new linear layer
                    hidden = (
                        head.in_features
                        if hasattr(head, "in_features")
                        else head.weight.size(1)
                    )
                    new_head = torch.nn.Linear(
                        hidden,
                        int(tok_vocab),
                        bias=(getattr(head, "bias", None) is not None),
                    )
                    with torch.no_grad():
                        ncopy = min(
                            getattr(head, "out_features", head.weight.size(0)),
                            int(tok_vocab),
                        )
                        new_head.weight[:ncopy].data.copy_(head.weight[:ncopy].data)
                        if (
                            getattr(head, "bias", None) is not None
                            and getattr(new_head, "bias", None) is not None
                        ):
                            new_head.bias[:ncopy].data.copy_(head.bias[:ncopy].data)
                    try:
                        new_head = new_head.to(head.weight.device)
                    except Exception as e:
                        logger.warning("Failed to move new text_head to device: %s", e)
                    setattr(tts.t3, "text_head", new_head)
        # show tokens for debugging
        try:
            sample_tokens = tts.tokenizer.text_to_tokens(punc_norm(args.text))
            logger.info("Sample token ids: %s", sample_tokens.squeeze(0).tolist())
        except Exception as e:
            logger.warning("Failed to show sample tokens: %s", e)
    except Exception as e:
        logger.warning("Pre-generation sanity checks failed: %s", e)

    # Move models to target device before generation
    try:
        try:
            tts.t3.to(device)
        except Exception as e:
            logger.warning("Failed to move t3 to device %s: %s", device, e)
        try:
            if hasattr(tts, "s3gen") and tts.s3gen is not None:
                tts.s3gen.to(device)
        except Exception as e:
            logger.warning("Failed to move s3gen to device %s: %s", device, e)
        try:
            if hasattr(tts, "ve") and tts.ve is not None:
                tts.ve.to(device)
        except Exception as e:
            logger.warning("Failed to move voice encoder to device %s: %s", device, e)
        tts.device = device
        # move conditionals to device as well
        try:
            if getattr(tts, "conds", None) is not None:
                try:
                    tts.conds = tts.conds.to(device)
                except Exception as e:
                    logger.warning("Failed to move conditionals to device: %s", e)
        except Exception:
            pass
        # quick param stats to detect zeroed/invalid weights
        try:
            stats = _module_param_stats(
                tts.t3, filters=["text_head", "text_emb"], max_items=8
            )
            logger.info("T3 param stats sample: %s", stats)
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed while moving models to device: %s", e)

    # Extra validation to avoid random-token outputs
    try:
        # sample tokens
        sample_tokens = tts.tokenizer.text_to_tokens(punc_norm(args.text)).squeeze(0)
        max_id = int(sample_tokens.max().item()) if sample_tokens.numel() else -1
        emb_size = None
        try:
            emb_size = getattr(tts.t3.text_emb, "num_embeddings", None)
            if emb_size is None and hasattr(tts.t3.text_emb, "weight"):
                emb_size = tts.t3.text_emb.weight.size(0)
        except Exception:
            emb_size = None
        head_out = None
        try:
            head_out = getattr(tts.t3.text_head, "out_features", None)
            if head_out is None and hasattr(tts.t3.text_head, "weight"):
                head_out = tts.t3.text_head.weight.size(0)
        except Exception:
            head_out = None
        logger.info(
            "Sample tokens max id=%s, emb_size=%s, head_out=%s",
            max_id,
            emb_size,
            head_out,
        )
        if emb_size is not None and max_id >= emb_size:
            raise RuntimeError(
                f"Tokenizer produced token id {max_id} >= text_emb size {emb_size}. Tokenizer and model vocab mismatch."
            )
        if head_out is not None and max_id >= head_out:
            raise RuntimeError(
                f"Tokenizer produced token id {max_id} >= text_head out_features {head_out}. Tokenizer and model vocab mismatch."
            )
    except Exception as e:
        logger.exception("Pre-generation validation failed: %s", e)
        raise

    logger.info("Generating...")
    try:
        wav = tts.generate(args.text)
    except Exception as e:
        msg = f"Generation failed: {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e

    try:
        if isinstance(wav, torch.Tensor):
            wav_out = wav.squeeze(0).cpu()
        else:
            import numpy as _np

            wav_out = torch.from_numpy(_np.asarray(wav)).squeeze(0)

        if wav_out.ndim == 1:
            wav_out = wav_out.unsqueeze(0)
        wav_out = wav_out.to(dtype=torch.float32)
        ta.save("tts_tr_example.wav", wav_out, sample_rate=tts.sr)
    except Exception as e:
        msg = f"Failed to save generated waveform: {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e

    logger.info("Saved tts_tr_example.wav")
