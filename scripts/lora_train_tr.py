import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torchinfo import summary
from dataset_utils import SimpleMetaDataset, collate_t3
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from safetensors.torch import load_file
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.tokenizers import EnTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_t3_from_ckpt(t3, ckpt_dir):
    try:
        state = load_file(Path(ckpt_dir) / "t3_cfg.safetensors")
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
        obj = t3
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
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--auto_offload", action="store_true")
    p.add_argument("--single_device_transformer", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Enable bnb 8-bit optimizer and bfloat16/autocast fast path (used in s3gen script)",
    )
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    en_tok = EnTokenizer(str(args.tokenizer))

    ds = SimpleMetaDataset(
        args.meta,
        cache_dir=args.cache_dir,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_t3(b, en_tok),
    )

    t3 = T3()
    load_t3_from_ckpt(t3, args.ckpt_dir)

    try:
        tok_vocab = len(en_tok.tokenizer.get_vocab())
    except Exception:
        tok_vocab = getattr(en_tok.tokenizer, "get_vocab_size", lambda: None)() or None

    try:
        if tok_vocab is not None and hasattr(t3, "text_emb"):
            cur_size = getattr(t3.text_emb, "num_embeddings", None)
            if cur_size is None and hasattr(t3.text_emb, "weight"):
                cur_size = t3.text_emb.weight.size(0)
            if cur_size is not None and tok_vocab > cur_size:
                old_emb = t3.text_emb
                new_emb = torch.nn.Embedding(tok_vocab, old_emb.embedding_dim)
                with torch.no_grad():
                    new_emb.weight[:cur_size].data.copy_(old_emb.weight[:cur_size].data)
                    try:
                        std = float(old_emb.weight.std().item())
                    except Exception:
                        std = 0.02
                    torch.nn.init.normal_(new_emb.weight[cur_size:], mean=0.0, std=std)
                t3.text_emb = new_emb
                if hasattr(t3, "text_head"):
                    old_head = t3.text_head
                    hidden = old_head.in_features
                    new_head = torch.nn.Linear(
                        hidden, tok_vocab, bias=getattr(old_head, "bias") is not None
                    )
                    with torch.no_grad():
                        ncopy = min(
                            getattr(old_head, "out_features", old_head.weight.size(0)),
                            tok_vocab,
                        )
                        new_head.weight[:ncopy].data.copy_(old_head.weight[:ncopy].data)
                        if tok_vocab > ncopy:
                            try:
                                std = float(old_head.weight.std().item())
                            except Exception:
                                std = 0.02
                            torch.nn.init.normal_(
                                new_head.weight[ncopy:], mean=0.0, std=std
                            )
                        if (
                            getattr(old_head, "bias", None) is not None
                            and new_head.bias is not None
                        ):
                            new_head.bias[:ncopy].data.copy_(old_head.bias[:ncopy].data)
                    t3.text_head = new_head
    except Exception:
        logger.exception("Failed while ensuring embeddings match tokenizer")
        raise

    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules="all-linear", inference_mode=False
    )

    t3.to("cpu")
    model = get_peft_model(t3, lora_config)

    base_wrapper = getattr(model, "base_model", model)
    for layer_name in ("text_emb", "text_head"):
        layer = getattr(base_wrapper, layer_name, None)
        if layer is not None:
            for pname, p in layer.named_parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise RuntimeError("No trainable parameters available for training")

    summary(model)

    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    tb_writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb"))
    global_step = 0
    model.train()
    model.to(device)

    grad_accum = max(1, int(args.grad_accum_steps))
    use_fp16 = getattr(args, "fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    accum_counter = 0
    accum_loss_speech = 0.0
    accum_loss_text = 0.0
    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"epoch {epoch}")

        for batch in loop:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            txt = batch["text_tokens"].long().to(device)

            B = txt.size(0)
            txt_len_padded = int(txt.size(1))
            txt_lens = torch.full((B,), txt_len_padded, dtype=torch.long, device=device)

            speech_tokens = batch.get("speech_tokens")
            speech_tokens = speech_tokens.long().to(device)

            speech_len_padded = int(speech_tokens.size(1))
            speech_lens = torch.full(
                (B,), speech_len_padded, dtype=torch.long, device=speech_tokens.device
            )

            spk_dim = (
                getattr(t3, "hp", None) and getattr(t3.hp, "speaker_embed_size", None)
            ) or 256
            speaker_emb = torch.zeros(B, spk_dim, device=device)
            emotion_tensor = torch.full((B, 1, 1), float(0.5), device=device)
            t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_tensor)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                lt, ls = model.loss(
                    t3_cond=t3_cond,
                    text_tokens=txt,
                    text_token_lens=txt_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_lens,
                )

                # combined loss for backward
                loss = lt + ls

                # scale for gradient accumulation
                loss = loss / float(grad_accum)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_counter += 1
                accum_loss_text += lt.item()
                accum_loss_speech += ls.item()
                if accum_counter % grad_accum == 0:
                    if scaler is not None:
                        scaler.unscale_(optim)
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad()

                    total_loss = accum_loss_text + accum_loss_speech
                    loop.set_postfix(
                        loss=total_loss / grad_accum,
                        loss_text=accum_loss_text / grad_accum,
                        loss_speech=accum_loss_speech / grad_accum,
                    )
                    tb_writer.add_scalar(
                        "train/loss",
                        total_loss / grad_accum,
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "train/loss_text",
                        accum_loss_text / grad_accum,
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "train/loss_speech",
                        accum_loss_speech / grad_accum,
                        global_step,
                    )
                    global_step += 1

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    peft_state = get_peft_model_state_dict(model)
    torch.save(peft_state, Path(args.out_dir) / "lora_t3_state.pt")
    base_updates = {
        "text_emb": t3.text_emb.state_dict(),
        "text_head": t3.text_head.state_dict(),
    }
    torch.save(base_updates, Path(args.out_dir) / "base_updates.pt")


if __name__ == "__main__":
    main()
