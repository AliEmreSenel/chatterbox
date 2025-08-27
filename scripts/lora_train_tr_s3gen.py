import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from dataset_utils import (SimpleMetaDataset, collate_s3gen,
                           recreate_missing_cache)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3Tokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_s3gen_from_ckpt(s3gen, ckpt_dir):
    try:
        from safetensors.torch import load_file
    except Exception:
        return
    try:
        state = load_file(Path(ckpt_dir) / "s3gen.safetensors")
    except Exception:
        return
    try:
        if "model" in state:
            state = state["model"][0]
    except Exception:
        pass
    import torch as _torch

    for k, v in list(state.items()):
        try:
            tensor = _torch.as_tensor(v)
        except Exception:
            continue
        parts = k.split(".")
        obj = s3gen
        ok = True
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if not ok:
            continue
        name = parts[-1]
        if not hasattr(obj, name):
            continue
        param = getattr(obj, name)
        try:
            if param.shape == tensor.shape:
                param.copy_(tensor)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--cache_dir", type=str, default="/mnt/data/cache/preprocess_tr_new_tok/"
    )
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Enable bnb 8-bit optimizer and bfloat16 autocast",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fast = bool(getattr(args, "fast", False))

    s3tok = S3Tokenizer()
    try:
        if torch.cuda.is_available() and getattr(args, "num_workers", 0) == 0:
            s3tok.to("cuda")
    except Exception:
        pass

    ds = SimpleMetaDataset(
        args.meta, s3tok, cache_dir=args.cache_dir, force_recompute=args.force_recompute
    )

    if args.cache_dir:
        try:
            recreate_missing_cache(args.cache_dir, ds.items, s3tok, batch_size=32)
        except Exception:
            logger.exception("Failed while recreating missing cache entries")

    s3gen = S3Gen()
    load_s3gen_from_ckpt(s3gen, args.ckpt_dir)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_s3gen(b, s3tok, s3gen),
    )

    # Dynamically determine reasonable target module name substrings for LoRA
    candidate_tokens = set()
    for name, mod in s3gen.named_modules():
        # consider last part of name and full name
        parts = name.split(".") if name else []
        for p in parts[-2:]:
            if any(x in p for x in ("proj", "linear", "conv", "embed", "mlp", "attn", "in_proj", "out_proj", "gate_proj", "up_proj", "down_proj")):
                candidate_tokens.add(p)
        # also check class name
        cls = mod.__class__.__name__.lower()
        if "conv" in cls:
            candidate_tokens.add("conv")
        if "linear" in cls or "dense" in cls:
            candidate_tokens.add("linear")
        if "projection" in cls or "proj" in cls:
            candidate_tokens.add("proj")
    # Fallback if nothing found
    if not candidate_tokens:
        candidate_tokens = {"proj", "linear", "conv"}
    target_modules = list(candidate_tokens)
    logger.info("Auto-detected target_modules for LoRA: %s", target_modules)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        inference_mode=False,
    )

    try:
        s3gen.to("cpu")
    except Exception:
        pass
    model = get_peft_model(s3gen, lora_config)

    for name, p in model.named_parameters():
        if "lora" in name.lower() and not p.requires_grad:
            p.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters available for training")

    optim = torch.optim.Adam(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb"))
    global_step = 0

    model.train()
    model.to(device)

    grad_accum = max(1, getattr(args, "grad_accum_steps", 1))
    MAX_TARGET_T = 288

    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")
        accum = 0
        for batch in loop:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }

            import torch.nn.functional as F

            from chatterbox.models.s3gen.utils.mask import make_pad_mask

            use_bf16 = use_fast and (device.type == "cuda")
            autocast_ctx = torch.cuda.amp.autocast(
                enabled=use_bf16, dtype=torch.bfloat16 if use_bf16 else None
            )

            flow = model.flow
            token = batch["speech_token"]
            token_len = batch["speech_token_len"]
            feat_orig = batch["speech_feat"]
            feat = feat_orig.transpose(1, 2).contiguous()
            feat_len = batch["speech_feat_len"]
            embedding = batch["embedding"]

            with autocast_ctx:
                emb = F.normalize(embedding, dim=1)
                emb = flow.spk_embed_affine_layer(emb)

                max_token_len = int(token.size(1))
                mask = (
                    (~make_pad_mask(token_len, max_len=max_token_len))
                    .float()
                    .unsqueeze(-1)
                    .to(device)
                )
                token_emb = flow.input_embedding(torch.clamp(token, min=0)) * mask

                h, h_lengths = flow.encoder(token_emb, token_len)
                h = flow.encoder_proj(h)

                mu = h.transpose(1, 2).contiguous()
                T_feat = feat.shape[2]
                T_mu = mu.shape[2]
                target_T = min(T_feat, T_mu, MAX_TARGET_T)

                mask_feat = (~make_pad_mask(feat_len)).to(h)
                if target_T < T_feat:
                    feat = feat[:, :, :target_T].contiguous()
                    if mask_feat.dim() == 3:
                        mask_feat = mask_feat[:, :, :target_T].contiguous()
                    else:
                        mask_feat = mask_feat[:, :target_T].contiguous()
                if target_T < T_mu:
                    mu = mu[:, :, :target_T].contiguous()

                conds = torch.zeros(
                    (
                        feat.size(0),
                        feat.size(1),
                        min(feat.shape[2], mu.shape[2], MAX_TARGET_T),
                    ),
                    device=feat.device,
                    dtype=feat.dtype,
                )
                loss, _ = flow.decoder.compute_loss(
                    feat, mask_feat.unsqueeze(1), mu, emb, cond=conds
                )
                if not torch.is_tensor(loss):
                    loss = torch.tensor(float(loss), device=device)
                loss = loss / float(grad_accum)

            loss.backward()
            accum += 1
            if accum % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optim.step()
                optim.zero_grad()
                report_loss = (loss * float(grad_accum)).detach()
                loop.set_postfix(loss=float(report_loss.cpu().item()))
                try:
                    tb_writer.add_scalar(
                        "train/loss", float(report_loss.cpu().item()), global_step
                    )
                except Exception:
                    logger.exception("Failed to write loss to TensorBoard")
                global_step += 1

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "s3gen_lora_finetuned.pt")


if __name__ == "__main__":
    main()
