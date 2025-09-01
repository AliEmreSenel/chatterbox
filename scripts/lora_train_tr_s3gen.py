import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from dataset_utils import SimpleMetaDataset, collate_s3gen
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from safetensors.torch import load_file
from chatterbox.models.s3gen import S3Gen


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
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SimpleMetaDataset(args.meta, cache_dir=args.cache_dir)

    s3gen = S3Gen()
    state = load_file(Path(args.ckpt_dir) / "s3gen.safetensors")
    res = s3gen.load_state_dict(state, strict=False)
    if getattr(res, "missing_keys", None):
        logger.warning("Missing keys in checkpoint: %s", res.missing_keys)
    if getattr(res, "unexpected_keys", None):
        logger.warning("Unexpected keys in checkpoint: %s", res.unexpected_keys)

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
    model = get_peft_model(s3gen.flow, lora_config)

    trainable_params = [
        p
        for p in (model.parameters() if model is not None else model.parameters())
        if p.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters available for training")

    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb"))
    global_step = 0

    model.to(device)
    model.train()

    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")
        for batch in loop:
            out = model(batch, device)
            loss = out["loss"]

            loss.backward()
            optim.step()
            optim.zero_grad()
            report_loss = loss.detach().cpu().item()
            loop.set_postfix(loss=report_loss)
            tb_writer.add_scalar("train/loss", report_loss, global_step)
            global_step += 1

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "s3gen_lora_finetuned.pt")


if __name__ == "__main__":
    main()
