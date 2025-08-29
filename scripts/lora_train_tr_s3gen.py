import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from dataset_utils import SimpleMetaDataset, collate_s3gen, recreate_missing_cache
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from safetensors.torch import load_file
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3Tokenizer
from torchinfo import summary
import bitsandbytes as bnb


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


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

    s3gen = S3Gen()
    state = load_file(Path(args.ckpt_dir) / "s3gen.safetensors")
    res = s3gen.load_state_dict(state, strict=False)
    if getattr(res, "missing_keys", None):
        logger.warning("Missing keys in checkpoint: %s", res.missing_keys)
    if getattr(res, "unexpected_keys", None):
        logger.warning("Unexpected keys in checkpoint: %s", res.unexpected_keys)

    if args.cache_dir:
        try:
            recreate_missing_cache(
                args.cache_dir, ds.items, s3tok, s3gen, batch_size=32
            )
        except Exception:
            logger.exception("Failed while recreating missing cache entries")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_s3gen(b),
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        inference_mode=False,
    )
    del s3tok
    model = get_peft_model(s3gen.flow, lora_config)

    base = getattr(model, "base_model", model)
    flow = base

    # encoder.forward -> checkpoint
    if hasattr(flow, "encoder") and hasattr(flow.encoder, "forward"):
        orig_enc = flow.encoder.forward

        def enc_forward(token_emb, token_len, *args, **kwargs):
            return checkpoint(
                lambda x: orig_enc(x, token_len, *args, **kwargs),
                token_emb,
                use_reentrant=False,
            )

        flow.encoder.forward = enc_forward

    # encoder_proj (Linear) -> checkpoint
    if hasattr(flow, "encoder_proj") and hasattr(flow.encoder_proj, "forward"):
        orig_proj = flow.encoder_proj.forward

        def proj_forward(x, *args, **kwargs):
            return checkpoint(
                lambda y: orig_proj(y, *args, **kwargs), x, use_reentrant=False
            )

        flow.encoder_proj.forward = proj_forward

    # spk_embed_affine_layer (Linear) -> checkpoint
    if hasattr(flow, "spk_embed_affine_layer") and hasattr(
        flow.spk_embed_affine_layer, "forward"
    ):
        orig_spk = flow.spk_embed_affine_layer.forward

        def spk_forward(x, *args, **kwargs):
            return checkpoint(
                lambda y: orig_spk(y, *args, **kwargs), x, use_reentrant=False
            )

        flow.spk_embed_affine_layer.forward = spk_forward

    # input_embedding (Embedding) -> checkpoint
    if hasattr(flow, "input_embedding") and hasattr(flow.input_embedding, "forward"):
        orig_emb = flow.input_embedding.forward

        def emb_forward(x, *args, **kwargs):
            return checkpoint(
                lambda y: orig_emb(y, *args, **kwargs), x, use_reentrant=False
            )

        flow.input_embedding.forward = emb_forward

    # decoder.compute_loss -> checkpoint
    if hasattr(flow, "decoder") and hasattr(flow.decoder, "compute_loss"):
        orig_dec_cl = flow.decoder.compute_loss

        def dec_compute_loss(*args, **kwargs):
            return checkpoint(
                lambda *targs: orig_dec_cl(*targs, **kwargs), *args, use_reentrant=False
            )

        flow.decoder.compute_loss = dec_compute_loss

    trainable_params = [
        p
        for p in (model.parameters() if model is not None else model.parameters())
        if p.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters available for training")

    optim = bnb.optim.Adam8bit(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb"))
    global_step = 0

    model.to(device)
    model.train()

    grad_accum = max(1, getattr(args, "grad_accum_steps", 1))

    # print model summary for flow-only model
    try:
        summary(model)
    except Exception:
        pass

    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")
        accum = 0
        for batch in loop:
            with autocast("cuda"):
                out = model(batch, device)
                loss = out.get("loss") if isinstance(out, dict) else out
                loss = loss / float(grad_accum)

            loss.backward()
            accum += 1
            if accum >= grad_accum:
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
                accum = 0
                global_step += 1

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "s3gen_lora_finetuned.pt")


if __name__ == "__main__":
    main()
